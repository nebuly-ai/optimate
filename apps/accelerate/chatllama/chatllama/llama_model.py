# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms
# of the GNU General Public License version 3.

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Union, Optional

import deepspeed
import torch
import torch.distributed
import torch.nn.functional as F
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from torch import nn
from transformers import AutoTokenizer

from llama import Tokenizer
from llama.generation import sample_top_p


class MyTokenizer:
    """Masked tokenizer of hugging face to be similar to the one of meta,
    just used for testing purposes.
    """

    def __init__(self, model_path: Optional[str] = None):

        if model_path is None:
            self.sp_model = AutoTokenizer.from_pretrained("gpt2")
        else:
            self.sp_model = AutoTokenizer.from_pretrained(model_path)

        self.n_words = self.sp_model.vocab_size
        self.bos_id = self.sp_model.bos_token_id
        self.eos_id = self.sp_model.eos_token_id
        self.pad_id = self.sp_model.eos_token_id

    def encode(
        self,
        s: str,
        bos: bool = True,
        eos: bool = True,
        truncation: bool = True,
    ) -> List[int]:
        output = self.sp_model.encode(s, truncation=truncation)
        t = list(output)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        input = torch.as_tensor(t)
        output = self.sp_model.decode(input)
        return output


class HFLikeTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        # assign attributes from real tokenizer to masked one
        self.pad_id = self.tokenizer.pad_id
        self.eos_id = self.tokenizer.eos_id
        self.bos_id = self.tokenizer.bos_id

        # mask attribute to be similar to hugging face
        self.eos_token_id = self.tokenizer.eos_id
        self.pad_token_id = self.tokenizer.pad_id

        # to match hugging face attribute
        self.pad_token_id = self.pad_id

    def create_sequence_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.where(
            tokens == self.tokenizer.pad_id,
            torch.zeros_like(tokens),
            torch.ones_like(tokens),
        )
        mask = torch.where(
            tokens == self.tokenizer.bos_id, torch.zeros_like(tokens), mask
        )
        mask = torch.where(
            tokens == self.tokenizer.eos_id, torch.zeros_like(tokens), mask
        )
        return mask

    def __call__(self, texts: Union[List[str], str], *args, **kwargs):
        if isinstance(texts, str):
            text = self.tokenizer.encode(texts, bos=True, eos=True)
            tokens = torch.tensor(text).long()
            mask = torch.ones_like(tokens)
        else:
            texts = [
                self.tokenizer.encode(text, bos=True, eos=True)
                for text in texts
            ]
            max_len = max(len(text) for text in texts)
            tokens = torch.full(
                (len(texts), max_len), self.tokenizer.pad_id
            ).long()
            for i, text in enumerate(texts):
                tokens[i, -len(text) :] = torch.tensor(  # noqa E203
                    text
                ).long()

            # TODO: decide how eos and bos should be handled - i need to mask
            # them? or not?
            mask = self.create_sequence_mask(tokens)
            for i in range(tokens.shape[0]):
                current_tokens = tokens[i, mask[i] == 1]
                tokens[
                    i, -len(current_tokens) - 1 : -1  # noqa E203
                ] = current_tokens
            mask = self.create_sequence_mask(tokens)

            # convert `pad_id` from -1 to 0, otherwise embedding will cause out
            # of bounds.
            tokens = torch.where(
                tokens == self.tokenizer.pad_id,
                torch.zeros_like(tokens),
                tokens,
            )
        output = {
            "input_ids": tokens,
            "attention_mask": mask,
        }
        return output

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


@dataclass
class ModelArgs:
    """This class is a modification of the ModelArgs class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference.
    """

    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    # defined later by tokenizer
    vocab_size: int = -1
    # make SwiGLU hidden layer size multiple of large power of 2
    multiple_of: int = 256
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024

    # added attributes
    froze_embeddings: bool = True
    use_fairscale: bool = True


class RMSNorm(torch.nn.Module):
    """This class is a modification of the RMSNorm class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """This class is a modification of the Attention class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        if args.use_fairscale:
            self.n_local_heads = (
                args.n_heads // fs_init.get_model_parallel_world_size()
            )
        else:
            self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        if args.use_fairscale:
            self.wq = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wk = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wv = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.wo = RowParallelLinear(
                args.n_heads * self.head_dim,
                args.dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
        else:
            self.wq = nn.Linear(
                args.dim, args.n_heads * self.head_dim, bias=False
            )
            self.wk = nn.Linear(
                args.dim, args.n_heads * self.head_dim, bias=False
            )
            self.wv = nn.Linear(
                args.dim, args.n_heads * self.head_dim, bias=False
            )
            self.wo = nn.Linear(
                args.n_heads * self.head_dim, args.dim, bias=False
            )

        self.dim_cache = (
            args.max_batch_size,
            args.max_seq_len,
            self.n_local_heads,
            self.head_dim,
        )
        self.cache_k = torch.zeros(self.dim_cache).cuda()

        self.cache_v = torch.zeros(self.dim_cache).cuda()

    def forward(
        self,
        x: torch.Tensor,
        kv_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_pos = 0  # Temporary

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Modified code to allow training, caching is not good for training
        if (cache_k is None and cache_v is not None) or (
            cache_k is not None and cache_v is None
        ):
            raise ValueError("cache_k is None while cache_v is not None")
        if cache_k is None:
            keys = xk
            values = xv
        else:
            cache_k.to(xk.device)
            cache_v.to(xv.device)
            cache_k[:bsz, start_pos : start_pos + seqlen] = xk  # noqa E203
            cache_v[:bsz, start_pos : start_pos + seqlen] = xv  # noqa E203
            keys = self.cache_k[:bsz, : start_pos + seqlen]  # noqa E203
            values = self.cache_v[:bsz, : start_pos + seqlen]  # noqa E203

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        if kv_mask is not None:
            scores = scores + kv_mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        if cache_k is None:
            return self.wo(output), None, None
        else:
            return self.wo(output), self.cache_k, self.cache_v


class FeedForward(nn.Module):
    """This class is a modification of the FeedForward class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference.
    """

    def __init__(
        self, dim: int, hidden_dim: int, multiple_of: int, use_fairscale: bool
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        if use_fairscale:
            self.w1 = ColumnParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            self.w2 = RowParallelLinear(
                hidden_dim,
                dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
            self.w3 = ColumnParallelLinear(
                dim,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """This class is a modification of the TransformerBlock class
    implemented in the LLaMA repo. The class has been modified for training,
    since the original one just supports inference.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            use_fairscale=args.use_fairscale,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_fairscale = args.use_fairscale

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # modified from orignal code to enable external cache
        attention_mask = attention_mask[:, None, :, :]
        if self.use_fairscale:
            attention_mask = attention_mask.expand(
                -1,
                self.n_heads // fs_init.get_model_parallel_world_size(),
                -1,
                -1,
            )
        else:
            attention_mask = attention_mask.expand(-1, self.n_heads, -1, -1)
        attn, cache_k, cache_v = self.attention.forward(
            self.attention_norm(x), attention_mask, freqs_cis, cache_k, cache_v
        )
        h = x + attn
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, cache_k, cache_v


class Transformer(nn.Module):
    """This class is a modification of the Transformer class implemented in
    the LLaMA repo. The class has been modified for training, since the
    original one just supports inference. The generate method was inspired by
    the generate function you can find in `llama.generation`.
    """

    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        if params.use_fairscale:
            self.n_local_heads = (
                params.n_heads // fs_init.get_model_parallel_world_size()
            )
        else:
            self.n_local_heads = params.n_heads

        self.head_dim = params.dim // params.n_heads
        dim = (
            params.max_batch_size,
            params.max_seq_len,
            self.n_local_heads,
            self.head_dim,
        )
        self.cache_k = [torch.zeros(dim) for _ in range(self.n_layers)]
        self.cache_v = [torch.zeros(dim) for _ in range(self.n_layers)]

        if params.use_fairscale:
            self.tok_embeddings = ParallelEmbedding(
                params.vocab_size, params.dim, init_method=lambda x: x
            )
        else:
            self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        if params.froze_embeddings:
            for param in self.tok_embeddings.parameters():
                param.requires_grad = False

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        if params.use_fairscale:
            self.output = ColumnParallelLinear(
                params.dim,
                params.vocab_size,
                bias=False,
                init_method=lambda x: x,
            )
        else:
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # TODO: How too modify this for training?
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = attention_mask.detach()
        logits = self._forward(tokens, attention_mask)
        return logits

    def _forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        # TEMPORARY FIX, need to understand how to manage the positioning
        # embedding and the batch size with the current padding and masking.
        start_pos = 1
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]  # noqa E203
        # mask has size (bsz, seqlen). It should be transformed in
        # (bsz, seqlen, seqlen)
        # if the mask is a boolean tensor, convert it to int
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.long()
        kv_mask = attention_mask[:, None, :].expand(_bsz, seqlen, seqlen)
        kv_mask = torch.tril(kv_mask, diagonal=0)
        kv_mask = 1 - kv_mask
        kv_mask = (
            torch.where(
                kv_mask == 1, kv_mask.new_tensor(-9223372036854775808), kv_mask
            )
            .detach()
            .long()
        )

        for i, layer in enumerate(self.layers):
            if not self.training:
                cache_k = self.cache_k[i]
                cache_v = self.cache_v[i]
                h, cache_k, cache_v = layer(
                    h, kv_mask, freqs_cis, cache_k, cache_v
                )
            else:
                h, _, _ = layer(h, kv_mask, freqs_cis)
            if not self.training:
                self.cache_k[i] = cache_k.detach()
                self.cache_v[i] = cache_v.detach()

        h = self.norm(h)
        output = self.output(h)
        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        no_repeat_ngram_size=None,
    ):
        generated_tokens = []
        for cur_pos in range(max_new_tokens):
            logits = self._forward(input_ids, attention_mask)[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token).unsqueeze(1)],
                dim=1,
            )
            generated_tokens.append(next_token)
        sequences = torch.concat(
            (input_ids, torch.stack(generated_tokens, dim=1)), dim=1
        )
        return sequences


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    print("local_rank:", local_rank, "world_size:", world_size)

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def setup_model_deepspeed() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    deepspeed.init_distributed()
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load_checkpoints(
    ckpt_dir: str, local_rank: int, world_size: int
) -> Tuple[dict, dict]:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(checkpoints), (
        f"Loading a checkpoint for MP={len(checkpoints)} but world "
        f"size is {world_size}"
    )
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    return checkpoint, params


def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    froze_embeddings: bool,
    use_fairscale: bool,
    max_batch_size: int = 32,
) -> Tuple[Transformer, HFLikeTokenizer]:

    checkpoint, params = load_checkpoints(ckpt_dir, local_rank, world_size)
    model_args: ModelArgs = ModelArgs(
        max_seq_len=1024, max_batch_size=max_batch_size, **params
    )
    model_args.froze_embeddings = froze_embeddings
    model_args.use_fairscale = use_fairscale
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    tokenizer = HFLikeTokenizer(tokenizer)

    return model, tokenizer


def load_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer(model_path=tokenizer_path)
    return tokenizer


def load_tokenizer_test(tokenizer_path: Optional[str] = None):
    tokenizer = MyTokenizer(model_path=tokenizer_path)
    return tokenizer


def load_model_test(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    froze_embeddings: bool,
    use_fairscale: bool,
    max_batch_size: int = 32,
) -> Tuple[Transformer, HFLikeTokenizer]:

    # test the model with hf tokenizer
    model_args = ModelArgs()
    model_args.froze_embeddings = froze_embeddings
    model_args.use_fairscale = use_fairscale
    tokenizer = MyTokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args).cuda()
    tokenizer = HFLikeTokenizer(tokenizer)

    return model, tokenizer
