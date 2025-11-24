import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_qkv_norm: bool = True,   # LN(x)
        use_qk_norm: bool = True,    # LN(q), LN(k)
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ----------- LN before qkv  -----------
        self.use_qkv_norm = use_qkv_norm
        if use_qkv_norm:
            self.qkv_norm = nn.LayerNorm(embed_dim)
        else:
            self.qkv_norm = nn.Identity()

        # ----------- qkv projection -----------
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)

        # ----------- per-head QK-Norm -----------
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # ----------- output projection -----------
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape

        # ----- Step 0: qkv input normalization -----
        x = self.qkv_norm(x)     # LN(embed_dim)

        # ----- Step 1: project qkv -----
        qkv = self.qkv(x)        # [B,N,3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # [B,H,N,Hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ----- Step 2: QK-Norm (optional) -----
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ----- Step 3: attention -----
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
