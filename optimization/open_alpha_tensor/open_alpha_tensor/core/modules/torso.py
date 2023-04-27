import torch

from open_alpha_tensor.core.modules.attention import AlphaMultiHeadAttention


class TorsoAttentiveModes(torch.nn.Module):
    def __init__(self, input_dim: int):
        # input_dim = c
        super().__init__()
        self.attention = AlphaMultiHeadAttention(
            input_dim,
            input_dim,
        )

    def forward(self, x1, x2, x3):
        # x1.size = x2.size = x3.size = (N, S, S, c)
        # where N is the batch size
        size = x1.shape[-2]
        input_list = [x1, x2, x3]
        for m1, m2 in [(0, 1), (2, 0), (1, 2)]:
            matrix = torch.cat([input_list[m1], input_list[m2]], dim=-2)
            # matrix_size = (N, S, 2S, c)
            out = self.attention(matrix, matrix)
            input_list[m1] = out[:, :, :size]
            input_list[m2] = out[:, :, size:]
        return input_list


class TorsoModel(torch.nn.Module):
    """Torso model of OpenAlphaTensor.

    It maps an input tensor of shape (N, T, S, S, S) to (N, 3S*S, c), where:

        N is the batch size;
        T is the context size (size of the history + 1);
        S is the number of elements in each matrix to be multiplied;
        c is the output dimensionality.
    """

    def __init__(
        self,
        scalars_size: int,
        input_size: int,
        tensor_length: int,
        out_size: int,
    ):
        # scalar_size = s
        # input_size = S
        # tensor_length = T
        # out_size = c
        super(TorsoModel, self).__init__()
        self.linears_1 = torch.nn.ModuleList(
            [
                torch.nn.Linear(scalars_size, input_size * input_size)
                for _ in range(3)
            ]
        )
        self.linears_2 = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_size * tensor_length + 1, out_size)
                for _ in range(3)
            ]
        )
        self.attentive_modes = torch.nn.ModuleList(
            [TorsoAttentiveModes(out_size) for _ in range(8)]
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor):
        # x.size = (N, T, S, S, S)
        # scalars.size = (N, s)
        batch_size = x.shape[0]
        S = x.shape[-1]
        T = x.shape[1]
        x1 = x.permute(0, 2, 3, 4, 1).reshape(batch_size, S, S, S * T)
        x2 = x.permute(0, 4, 2, 3, 1).reshape(batch_size, S, S, S * T)
        x3 = x.permute(0, 3, 4, 2, 1).reshape(batch_size, S, S, S * T)
        input_list = [x1, x2, x3]
        for i in range(3):
            temp = self.linears_1[i](scalars).reshape(batch_size, S, S, 1)
            input_list[i] = torch.cat([input_list[i], temp], dim=-1)
            input_list[i] = self.linears_2[i](input_list[i])
        x1, x2, x3 = input_list
        for layer in self.attentive_modes:
            x1, x2, x3 = layer(x1, x2, x3)
        return torch.stack([x1, x2, x3], dim=2).reshape(
            batch_size, 3 * S * S, -1
        )
