import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    Feed-forward layer using the SwiGLU activation:
        hidden = silu(x @ W1_a) * (x @ W1_b)
        out    = hidden @ W2

    This is a lightweight, standalone implementation appropriate for
    ViT blocks in DINO-style self-supervised training.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
    ):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim

        # First linear layer produces two chunks: (a, b)
        self.fc12 = nn.Linear(in_dim, 2 * hidden_dim)

        # Second linear layer maps hidden_dim -> out_dim
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x12: [B, N, 2*hidden]
        x12 = self.fc12(x)

        # Split into two equal parts
        a, b = x12.chunk(2, dim=-1)

        # SwiGLU activation
        hidden = F.silu(a) * b

        # Final projection
        return self.fc3(hidden)
