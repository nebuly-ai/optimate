try:
    from transformers import (
        PreTrainedModel,
        CLIPTextModel,
    )
    from transformers.tokenization_utils import PreTrainedTokenizer
except ImportError:
    # add placeholders for function definition
    PreTrainedModel = None
    CLIPTextModel = None
    PreTrainedTokenizer = None
