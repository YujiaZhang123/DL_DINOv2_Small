import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Standard 2-layer feed-forward MLP used in ViT blocks.
    fc1 → GELU → dropout → fc2 → dropout
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim

        # First linear: in_dim -> hidden_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)

        # Activation layer (GELU by default)
        self.act = act_layer()

        # Dropout
        self.drop = nn.Dropout(drop)

        # Second linear: hidden_dim -> out_dim
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
