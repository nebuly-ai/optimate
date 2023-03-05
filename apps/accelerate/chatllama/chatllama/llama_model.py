import json
import os
from pathlib import Path
from typing import Tuple, List, Union

import deepspeed
import torch.distributed
import torch.nn as nn
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    ColumnParallelLinear,
)

# All code imported from llama is owned by Meta and used here with no
# modification
from llama import ModelArgs, Tokenizer
from llama.generation import sample_top_p
from llama.model import TransformerBlock, RMSNorm, precompute_freqs_cis


class HFLikeTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts: Union[List[str], str], *args, **kwargs):
        if isinstance(texts, str):
            text = self.tokenizer.encode(texts, bos=True, eos=True)
            tokens = torch.tensor(text).cuda().long()
        else:
            texts = [
                self.tokenizer.encode(text, bos=True, eos=True)
                for text in texts
            ]
            max_len = max(len(text) for text in texts)
            tokens = (
                torch.full((len(texts), max_len), self.tokenizer.pad_id)
                .cuda()
                .long()
            )
            for i, text in enumerate(texts):
                tokens[i, : len(text)] = torch.tensor(text).cuda().long()
        output = {
            "input_ids": tokens,
            "attention_mask": (tokens != self.tokenizer.pad_id).long(),
        }
        return output

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


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

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor):
        start_pos = int(torch.argmax(attention_mask.detach(), dim=-1).item())
        logits = self._forward(tokens, start_pos)
        return logits

    def _forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]  # noqa E203

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float = 1.0,
    ):
        prompt_size = inputs.shape[1]
        total_len = min(self.params.max_seq_len, max_length + prompt_size)
        start_pos = prompt_size  # We assume left padding
        prev_pos = 0
        generated_tokens = []
        for cur_pos in range(start_pos, total_len):
            logits = self._forward(inputs[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            generated_tokens.append(next_token)
            prev_pos = cur_pos
        return torch.stack(generated_tokens, dim=1)


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

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
    max_batch_size: int = 32,
) -> Tuple[Transformer, HFLikeTokenizer]:
    checkpoint, params = load_checkpoints(ckpt_dir, local_rank, world_size)
    model_args: ModelArgs = ModelArgs(
        max_seq_len=1024, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    tokenizer = HFLikeTokenizer(tokenizer)
    return model, tokenizer
