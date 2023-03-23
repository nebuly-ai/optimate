from nebullvm.optional_modules.dummy import DummyClass

try:
    from transformers import PreTrainedModel, CLIPTextModel, CLIPTokenizer
    from transformers.tokenization_utils import PreTrainedTokenizer
except ImportError:
    # add placeholders for function definition
    PreTrainedModel = DummyClass
    CLIPTextModel = DummyClass
    CLIPTokenizer = DummyClass
    PreTrainedTokenizer = DummyClass
