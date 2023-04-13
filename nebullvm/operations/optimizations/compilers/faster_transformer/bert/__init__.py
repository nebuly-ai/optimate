import os

from nebullvm.operations.optimizations.compilers.faster_transformer.bert.modeling_bert import (  # noqa: E501
    BertModel as FasterBertModel,
)
from nebullvm.operations.optimizations.compilers.faster_transformer.bert.modeling_bert import (  # noqa: E501
    CustomEncoder,
    EncoderWeights,
)
from nebullvm.operations.optimizations.compilers.utils import (
    get_faster_transformer_repo_path,
)
from nebullvm.optional_modules.huggingface import BertModel as HFBertModel
from nebullvm.optional_modules.torch import torch


default_lib_path = str(
    get_faster_transformer_repo_path()
    / "build"
    / "lib"
    / "libth_transformer.so"
)


def swap_bert_encoder(model, data_type, lib_path, remove_padding=False):
    """
    Replace the encoder of the model with a custom encoder
    that uses the Faster Transformer library.
    """
    weights = EncoderWeights(
        model.config.num_hidden_layers,
        model.config.hidden_size,
        model.state_dict(),
    )
    weights.to_cuda()
    if data_type == "fp16":
        weights.to_half()
    elif data_type == "bf16":
        weights.to_bfloat16()
    lib_path = os.path.abspath(lib_path)
    enc = CustomEncoder(
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.hidden_size // model.config.num_attention_heads,
        weights,
        remove_padding=remove_padding,
        path=lib_path,
    )
    enc_ = torch.jit.script(enc)
    model.replace_encoder(enc_)


def swap_model(
    model: HFBertModel, data_type, lib_path, remove_padding=False
) -> FasterBertModel:
    # bert model need some custom code to call the custom encoder
    # so we need to use custom bert class
    new_model = FasterBertModel(model.config)
    new_model.load_state_dict(model.state_dict())
    swap_bert_encoder(new_model, data_type, lib_path, remove_padding)
    return new_model


def detect_and_swap_bert_model(
    model, data_type, lib_path=default_lib_path, remove_padding=False
):
    if type(model) == HFBertModel:
        model = swap_model(model, data_type, lib_path, remove_padding)
    if hasattr(model, "bert") and type(model.bert) == HFBertModel:
        model.bert = swap_model(
            model.bert, data_type, lib_path, remove_padding
        )
    return model
