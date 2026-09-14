# StreamingBDH: inference over the synaptic state of a trained BDH checkpoint.
# The model is fed one position at a time; nothing is ever re-read.
#
# Design:
#   - state: one N x D matrix per (layer pass, head): shape (L, nh, N, D)
#   - per new position, per layer pass: sparse activation -> rotary-encode at its own position ->
#       READ  attention output = encoded activation @ state[pass]
#       WRITE state[pass] += encoded activation (outer) residual vector, after the read
#   - the gate uses the un-encoded activation; residual and layer norm as in bdh.py
#   - no training, no approximation: outputs must match recompute exactly
import os

import bdh
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


class StreamingBDH:
    def __init__(self, ckpt_path=os.path.join(HERE, "checkpoints", "bdh-100m-bytes.pt"),
                 device="cuda", dtype=torch.float32, window=None):
        """window=None: the state never forgets (every position stays written).
        window=W: sliding window. When a position falls W positions behind, its own
        outer products are subtracted from every layer pass's state. This is exact at
        the first pass and approximate at deeper passes, because later positions'
        deeper-pass residual vectors were computed from reads that included this
        position, and those traces stay written (the same non-equivalence as a
        transformer's sliding-window KV cache)."""
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = bdh.BDHConfig(**ck["config"])
        self.model = bdh.BDH(self.cfg)
        self.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = self.model.to(self.device, dtype).eval()

        C = self.cfg
        self.L = C.n_layer
        self.nh = C.n_head
        self.D = C.n_embd
        self.N = C.mlp_internal_dim_multiplier * C.n_embd // C.n_head
        self.window = window
        self.reset()

    def reset(self):
        """Zero every state matrix and reset the position to 0."""
        self.grids = torch.zeros(self.L, self.nh, self.N, self.D,
                                 device=self.device, dtype=self.dtype)
        self.pos = 0
        # ring buffer of each level's INPUT board vector per position; enough to
        # rebuild (chord -> stamp) an old etch exactly when it must be evicted
        self.xbuf = (torch.zeros(self.window, self.L, self.D, device=self.device,
                                 dtype=self.dtype) if self.window else None)

    @torch.no_grad()
    def _evict(self):
        """Subtract the position that just fell out of the window from every state matrix."""
        m = self.model
        old = self.pos - self.window
        slot = old % self.window
        old_phases = old * m.attn.freqs
        for level in range(self.L):
            x_old = self.xbuf[slot, level]                           # (D,)
            chord_old = torch.relu(x_old.view(1, 1, 1, self.D) @ m.encoder)
            stamped_old = m.attn.rope(old_phases, chord_old)         # (1,nh,1,N)
            self.grids[level] -= stamped_old[0, :, 0, :].unsqueeze(-1) * x_old

    @torch.no_grad()
    def step(self, byte_id):
        """Feed one position; return the logits for the next one."""
        m = self.model
        # one byte -> its board vector, shaped like a 1-token batch (1,1,1,D)
        x = m.embed(torch.tensor([[byte_id]], device=self.device)).unsqueeze(1)
        x = m.ln(x)

        # this token's clock angles: position x every clock's rate  (1,1,1,N)
        phases = self.pos * m.attn.freqs
        if self.window and self.pos >= self.window:
            self._evict()

        for level in range(self.L):
            chord = torch.relu(x @ m.encoder)          # (1, nh, 1, N)
            stamped = m.attn.rope(phases, chord)       # dated chord

            # read: (1, nh, 1, N) @ (nh, N, D) -> (1, nh, 1, D)
            mail = stamped @ self.grids[level]
            # write, after the read: outer product of the encoded activation and the residual vector
            self.grids[level] += stamped[0, :, 0, :].unsqueeze(-1) * x[0, 0, 0, :]
            if self.window:
                self.xbuf[self.pos % self.window, level] = x[0, 0, 0, :]

            yKV = m.ln(mail)
            y_sparse = torch.relu(yKV @ m.encoder_v)
            gated = chord * y_sparse                   # veto: UNstamped chord
            yMLP = gated.transpose(1, 2).reshape(1, 1, 1, self.N * self.nh) \
                        @ m.decoder
            x = m.ln(x + m.ln(yMLP))

        self.pos += 1
        return (x.view(1, self.D) @ m.lm_head)[0].float()

    @torch.no_grad()
    def generate(self, prompt_bytes, n_new, temperature=0.8, top_k=40, seed=0):
        """Convenience: prime with a prompt, then sample n_new bytes."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        logits = None
        for b in prompt_bytes:
            logits = self.step(b)
        out = []
        for _ in range(n_new):
            lg = logits / temperature
            v, _ = torch.topk(lg, top_k)
            lg[lg < v[-1]] = float("-inf")
            probs = torch.softmax(lg, dim=-1).cpu()
            b = int(torch.multinomial(probs, 1, generator=g))
            out.append(b)
            logits = self.step(b)
        return bytes(out)
