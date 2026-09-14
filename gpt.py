# Minimal decoder-only transformer (nanoGPT-style) as a parameter-matched
# baseline for the BDH scaling series. Same interface as bdh.BDH:
#   GPT(GPTConfig(...)).forward(idx, targets=None) -> (logits, loss)
# Rotary positions (so the context-span comparison with BDH's clocks is fair),
# pre-LN blocks, causal MHA via F.scaled_dot_product_attention, 4x GELU MLP,
# final LN, UNTIED lm_head, no biases.
#   python gpt.py            -> prints exact param counts for candidate configs
import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class GPTConfig:
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    vocab_size: int = 256
    dropout: float = 0.1
    block_size: int = 512


def rope_cos_sin(T, hd, device, dtype, theta=10000.0):
    """Standard RoPE tables for head dim hd: (T, hd/2) cos and sin."""
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    ang = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: (B, nh, T, hd); rotate pairs (even, odd)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.hd = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        cos, sin = rope_cos_sin(T, self.hd, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # untied
        self.apply(self._init_weights)
        for n, p in self.named_parameters():          # GPT-2 residual-projection scaling
            if n.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence {T} > block_size {self.config.block_size}"
        x = self.drop(self.embed(idx))
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


def n_params(cfg):
    return sum(p.numel() for p in GPT(cfg).parameters())


if __name__ == "__main__":
    import bdh
    ref = sum(p.numel() for p in bdh.BDH(bdh.BDHConfig(n_layer=6, n_embd=256, n_head=4,
                                                       mlp_internal_dim_multiplier=128,
                                                       vocab_size=256)).parameters())
    print(f"BDH 25M reference (D=256, bytes): {ref:,}")
    for L, D in ((8, 512), (6, 576), (12, 448), (4, 704)):
        n = n_params(GPTConfig(n_layer=L, n_embd=D, n_head=max(1, D // 64)))
        print(f"GPT n_layer={L:2d} n_embd={D:3d} n_head={max(1, D // 64)}: {n:,}  ({(n/ref-1)*100:+.1f}% vs BDH)")
