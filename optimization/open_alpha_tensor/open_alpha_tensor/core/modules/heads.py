import math

import torch
import torch.nn.functional as F

from open_alpha_tensor.core.modules.attention import AlphaMultiHeadAttention


class PositionEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return x


class PolicyHeadDoubleAttention(torch.nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_heads: int,
        n_feat: int,
        emb_size: int,
        emb_dim: int,
    ):
        super().__init__()
        d_model = n_feat * n_heads
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.attention1 = AlphaMultiHeadAttention(d_model, d_model)
        self.drop1 = torch.nn.Dropout()
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.attention2 = AlphaMultiHeadAttention(d_model, emb_dim)
        self.drop2 = torch.nn.Dropout()

    def forward(self, x: torch.Tensor, e: torch.Tensor):
        x = self.layer_norm1(x)
        c = self.attention1(x, x, mask=True)
        c = self.drop1(c)
        x = x + c
        x = self.layer_norm2(x)
        c = self.attention2(x, e, mask=False)
        c = self.drop2(c)
        x = x + c
        return x


class PolicyHeadCore(torch.nn.Module):
    def __init__(
        self,
        emb_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_feat: int = 64,
        n_heads: int = 32,
        n_layers: int = 2,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_logits, n_feat * n_heads)
        self.position_encoding = PositionEncoding(n_feat * n_heads)
        self.decoders = torch.nn.ModuleList(
            [
                PolicyHeadDoubleAttention(
                    n_steps, n_heads, n_feat, emb_size, emb_dim
                )
                for _ in range(n_layers)
            ]
        )
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n_feat * n_heads, n_logits)

    def forward(self, a: torch.Tensor, e: torch.Tensor):
        x = self.position_encoding(self.embedding(a))
        for layer in self.decoders:
            x = layer(x, e)
        o = self.linear2(self.relu(x))
        return o, x


def sample_from_logits(a):
    # returns a sampled element and the associated probability
    # since cross entropy is run during training we expect logits
    # to be probabilities yet.
    probs = torch.cumsum(F.softmax(a, dim=-1), dim=-1)
    random_vals = torch.rand(probs.shape[0]).unsqueeze(-1).to(a.device)
    n_classes = a.shape[-1]
    new_a_idx = torch.argmax(1.0 * (probs > random_vals), dim=-1)
    index_bias = torch.arange(0, len(new_a_idx)).to(a.device) * n_classes
    probs = torch.take(probs, new_a_idx + index_bias)
    # new_a = F.one_hot(new_a_idx, n_classes)
    return new_a_idx, probs


class PolicyHead(torch.nn.Module):
    def __init__(
        self,
        emb_size: int,
        emb_dim: int,
        n_steps: int,
        n_logits: int,
        n_samples: int,
    ):
        super().__init__()
        self.n_logits = n_logits
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.core = PolicyHeadCore(emb_size, emb_dim, n_steps, n_logits)

    def _train_forward(self, e: torch.Tensor, g: torch.Tensor):
        # e is the embedding, shape = (N, m, c)
        # g represents the previous actions, when training it represents the
        # list of correct actions, thus we need to shift them (since we do not
        # want to consider also the latest, correct action when predicting).
        # g has shape (N, N_steps) and it is a one-hot encoding of N_logits
        g = torch.roll(g, shifts=-1, dims=1)
        # the first raw will have attention zero during training
        # g = F.one_hot(g, self.n_logits).float()
        o, z = self.core(g, e)
        return o, z[:, 0]

    def _eval_forward(self, e: torch.Tensor):
        bs = e.shape[0]
        future_g = (
            torch.zeros((bs, self.n_samples, self.n_steps)).long().to(e.device)
        )
        ps = torch.ones((bs, self.n_samples)).to(e.device)
        e = e.unsqueeze(1).repeat(1, self.n_samples, 1, 1)

        future_g = future_g.view(-1, self.n_steps)
        ps = ps.view(-1)
        e = e.view(-1, e.shape[-2], e.shape[-1])
        for i in range(self.n_steps):
            o_s, z_s = self.core(future_g[:, : i + 1], e)
            future_g[:, i], p_i = sample_from_logits(o_s[:, i])
            ps *= p_i
        future_g = future_g.view(bs, self.n_samples, self.n_steps)
        ps = ps.view(bs, self.n_samples)
        return (
            future_g,
            ps,
            z_s[:, 0].view(bs, self.n_samples, *z_s.shape[2:]).mean(1),
        )

    def forward(self, e: torch.Tensor, g: torch.Tensor = None):
        if g is None:
            return self._eval_forward(e)
        return self._train_forward(e, g)


class ValueHeadCore(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.relu(self.linear(x))


class ValueHead(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int = 512, output_size: int = 8
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *(
                [ValueHeadCore(input_size, hidden_size)]
                + [ValueHeadCore(hidden_size, hidden_size)] * 2
            )
        )
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        return self.linear(self.layers(x))
