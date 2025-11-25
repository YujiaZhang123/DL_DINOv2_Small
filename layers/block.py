import torch
import torch.nn as nn
from layers.attention import MultiHeadAttention
from layers.drop_path import DropPath
from layers.swiglu_ffn import SwiGLUFFN


class TransformerBlock(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.035,
        use_layerscale: bool = True,
    ):
        super().__init__()

        # --- attention branch ---
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- feed-forward branch ---
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # -------- LayerScale --------
        if use_layerscale:
            self.gamma1 = nn.Parameter(1e-5 * torch.ones(embed_dim))
            self.gamma2 = nn.Parameter(1e-5 * torch.ones(embed_dim))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.gamma1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x)))
            x = x + self.drop_path2(self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma1 * self.attn(self.norm1(x)))
            x = x + self.drop_path2(self.gamma2 * self.ffn(self.norm2(x)))

        return x
