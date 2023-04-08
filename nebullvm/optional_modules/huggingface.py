try:
    from transformers import (
        PreTrainedModel,
    )
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.models.bert.modeling_bert import BertModel

except ImportError:
    # add placeholders for function definition
    PreTrainedModel = None
    PreTrainedTokenizer = None
    BertModel = None
