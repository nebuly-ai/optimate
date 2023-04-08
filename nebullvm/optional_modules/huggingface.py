try:
    from transformers import (
        PreTrainedModel,
    )
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.models.bert.modeling_bert import (
        BertModel,
        BertEmbeddings,
        BertEncoder,
        BertPooler,
        BertPreTrainedModel,
    )
    from transformers import BertConfig
except ImportError:
    # add placeholders for function definition
    PreTrainedModel = None
    PreTrainedTokenizer = None
    BertModel = None
    BertEmbeddings = None
    BertEncoder = None
    BertPooler = None
    BertPreTrainedModel = None
    BertConfig = None
