import torch
from torch.nn import functional as F


class AttentionHead(torch.nn.Module):
    def __init__(self, x_size: int, y_size: int, proj_dim: int):
        # x_size = N_x
        # y_size = N_y
        super(AttentionHead, self).__init__()
        self.proj_dim = proj_dim
        self.proj_dim_isqrt = 1 / torch.sqrt(torch.tensor(proj_dim))
        self.queries_proj_layer = torch.nn.Linear(x_size, proj_dim)
        self.keys_proj_layer = torch.nn.Linear(y_size, proj_dim)
        self.values_proj_layer = torch.nn.Linear(y_size, proj_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: bool = False):
        queries = self.queries_proj_layer(x)
        keys = self.keys_proj_layer(y)
        values = self.values_proj_layer(y)
        attention = F.softmax(
            torch.matmul(queries, keys.transpose(-2, -1))
            * self.proj_dim_isqrt,
            dim=-1,
        )
        if mask:
            attention = torch.triu(attention, diagonal=1)
        output = torch.matmul(attention, values)
        return output


class AttentionDenseBlock(torch.nn.Module):
    def __init__(self, inner_size: int, multiplier: int = 4):
        super().__init__()
        self.norm = torch.nn.LayerNorm(inner_size)
        self.linear = torch.nn.Linear(inner_size, inner_size * multiplier)
        self.activation = torch.nn.GELU()
        self.linear_final = torch.nn.Linear(
            inner_size * multiplier, inner_size
        )

    def forward(self, x: torch.Tensor):
        x_temp = self.activation(self.linear(self.norm(x)))
        return x + self.linear_final(x_temp)


class AlphaMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        proj_dim: int = 32,
        n_heads: int = 16,
        multiplier: int = 4,
    ):
        # x_dim = size of the last dimension of x
        # y_dim = size of the last dimension of y
        super().__init__()
        self.norm_layer_x = torch.nn.LayerNorm(x_dim)
        self.norm_layer_y = torch.nn.LayerNorm(y_dim)
        self.module_list = torch.nn.ModuleList(
            [AttentionHead(x_dim, y_dim, proj_dim) for _ in range(n_heads)]
        )
        self.linear = torch.nn.Linear(n_heads * proj_dim, x_dim)

        self.dense = AttentionDenseBlock(x_dim, multiplier)

    def forward(
        self, x: torch.nn.Module, y: torch.nn.Module, mask: bool = False
    ):
        # x.size = (Nx, c1), y.size = (Ny, c2)
        x_norm = self.norm_layer_x(x)
        y_norm = self.norm_layer_y(y)
        temp = torch.cat(
            [layer(x_norm, y_norm, mask) for layer in self.module_list], dim=-1
        )
        x = x + self.linear(temp)
        return self.dense(x)
