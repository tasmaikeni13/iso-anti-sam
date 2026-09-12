import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [batch, n_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class CausalAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, seq_len: int = 2048):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # PyTorch FlashAttention via scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, seq_len: int = 2048):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalAttention(dim, n_heads, seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class CausalLMTransformer(nn.Module):
    """
    Modern Decoder-Only Causal Language Model.
    Architecture: RoPE, RMSNorm, SwiGLU, FlashAttention (SDPA).
    """
    def __init__(self, vocab_size: int = 50257, dim: int = 768, n_layers: int = 12,
                 n_heads: int = 12, hidden_dim: int = 2048, max_seq_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim, seq_len=max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def get_num_params(self, exclude_embeddings: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

def create_model_125m(vocab_size: int = 50257, max_seq_len: int = 2048) -> CausalLMTransformer:
    """125M-parameter configuration (~125M active parameters)."""
    return CausalLMTransformer(
        vocab_size=vocab_size,
        dim=768,
        n_layers=12,
        n_heads=12,
        hidden_dim=2048,
        max_seq_len=max_seq_len
    )

def create_model_350m(vocab_size: int = 50257, max_seq_len: int = 2048) -> CausalLMTransformer:
    """350M-parameter configuration (~350M active parameters)."""
    return CausalLMTransformer(
        vocab_size=vocab_size,
        dim=1024,
        n_layers=24,
        n_heads=16,
        hidden_dim=2816,
        max_seq_len=max_seq_len
    )
