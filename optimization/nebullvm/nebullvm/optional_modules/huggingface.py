from nebullvm.optional_modules.dummy import DummyClass

try:
    from transformers import PreTrainedModel, CLIPTextModel, CLIPTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.models.bert.modeling_bert import (
        BertModel,
        BertEmbeddings,
        BertEncoder,
        BertPooler,
        BertPreTrainedModel,
    )
    from transformers import BertConfig, GPT2Tokenizer, GPT2LMHeadModel
except ImportError:
    # add placeholders for function definition
    PreTrainedModel = DummyClass
    CLIPTextModel = DummyClass
    CLIPTokenizer = DummyClass
    PreTrainedTokenizer = DummyClass
    BertModel = DummyClass
    BertEmbeddings = DummyClass
    BertEncoder = DummyClass
    BertPooler = DummyClass
    BertPreTrainedModel = DummyClass
    BertConfig = DummyClass
    GPT2Tokenizer = DummyClass
    GPT2LMHeadModel = DummyClass
