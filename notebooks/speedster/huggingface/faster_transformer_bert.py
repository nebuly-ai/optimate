# %%
import logging
import random
import time

import speedster
import torch
from speedster import optimize_model

# %%
from nebullvm.operations.optimizations.compilers.faster_transformer.bert import (  # noqa: E501
    detect_and_swap_bert_model,
)

# %%
from nebullvm.operations.optimizations.compilers.utils import (
    get_faster_transformer_repo_path,
)
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification as HFBertForSequenceClassification,
)

# %%
print(speedster.__file__)
lib_path = str(
    get_faster_transformer_repo_path()
    / "build"
    / "lib"
    / "libth_transformer.so"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# https://huggingface.co/bert-base-cased-finetuned-mrpc


# %%
def prepare_examples(tokenizer, len_dataset=1000):
    sentences = [
        "Mars is the fourth planet from the Sun.",
        "has a crust primarily composed of elements",
        "However, it is unknown",
        "can be viewed from Earth",
        "It was the Romans",
    ]
    texts = []
    for _ in range(len_dataset):
        n_times = random.randint(1, 30)
        texts.append(
            " ".join(random.choice(sentences) for _ in range(n_times))
        )
    encoded_inputs = [
        tokenizer(text, return_tensors="pt", truncation=True).to(device)
        for text in texts
    ]
    len(encoded_inputs), encoded_inputs[0].keys()
    fake_input_id = torch.LongTensor(per_gpu_eval_batch_size, max_seq_length)
    fake_input_id.fill_(1)
    fake_input_id = fake_input_id.to(device)
    fake_mask = torch.ones(per_gpu_eval_batch_size, max_seq_length).to(device)
    fake_type_id = fake_input_id.clone().detach()
    if data_type == "fp16":
        fake_mask = fake_mask.half()
    elif data_type == "bf16":
        fake_mask = fake_mask.bfloat16()
    return encoded_inputs, fake_input_id, fake_mask, fake_type_id


# %%
logger = logging.getLogger(__name__)
use_ths = use_torchscript = False
remove_padding = False
data_type = "fp16"  # "fp32", "fp16", "bf16"

per_gpu_eval_batch_size = 1
max_seq_length = 128
model_name_or_path = "bert-base-cased-finetuned-mrpc"


model = HFBertForSequenceClassification.from_pretrained(
    model_name_or_path, torchscript=True
)
model.eval().to(device)
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
encoded_inputs, fake_input_id, fake_mask, fake_type_id = prepare_examples(
    tokenizer
)


def optimize_no_trace(model, data_type="fp16"):
    model = detect_and_swap_bert_model(
        model, data_type="fp16", lib_path=lib_path, remove_padding=False
    )
    if data_type == "fp16":
        logger.info("Use fp16")
        model.half()
    elif data_type == "bf16":
        logger.info("Use bf16")
        model.bfloat16()
    return model.to(device)


def optimize_with_trace(
    model, data_type, per_gpu_eval_batch_size, max_seq_length
):
    model = optimize_no_trace(model, data_type)
    logger.info("Use TorchScript mode")
    fake_input_id = torch.LongTensor(per_gpu_eval_batch_size, max_seq_length)
    fake_input_id.fill_(1)
    fake_input_id = fake_input_id.to(device)
    fake_mask = torch.ones(per_gpu_eval_batch_size, max_seq_length).to(device)
    fake_type_id = fake_input_id.clone().detach()
    if data_type == "fp16":
        fake_mask = fake_mask.half()
    elif data_type == "bf16":
        fake_mask = fake_mask.bfloat16()
    model.eval()
    with torch.no_grad():
        model_ = torch.jit.trace(
            model, (fake_input_id, fake_mask, fake_type_id)
        )
    return model_


def benchmark(model, model_desc="original BERT"):
    times = []

    # Warmup for 30 iterations
    for encoded_input in encoded_inputs[:30]:
        with torch.no_grad():
            _ = model(**encoded_input)

    # Benchmark
    for encoded_input in encoded_inputs:
        st = time.perf_counter()
        with torch.no_grad():
            _ = model(**encoded_input)
        times.append(time.perf_counter() - st)
    original_model_time = sum(times) / len(times) * 1000
    print(f"Average response time for {model_desc}: {original_model_time} ms")


print(f"{encoded_inputs[0].keys()}")


benchmark(model, "BERT")
benchmark(model, "BERT")
data_type = "fp16"  # "fp32", "fp16", "bf16
per_gpu_eval_batch_size = 1
max_seq_length = 128
faster_model = optimize_no_trace(model, data_type)
benchmark(faster_model, "faster BERT (no metric drop)")
# Average response time for BERT: 4.741025467636064 ms
# Average response time for BERT: 4.686204055091366 ms

fastest_model = optimize_with_trace(
    model, data_type, per_gpu_eval_batch_size, max_seq_length
)

benchmark(fastest_model, "fastest BERT (no metric drop)")
# Average response time for faster BERT (no metric drop): 1.5583459960762411 ms # noqa: E501


# the above operations modifies `model` in-place
# so we need reload a fresh one to test speedster
model = HFBertForSequenceClassification.from_pretrained(
    model_name_or_path, torchscript=True
)
# Average response time for fastest BERT (no metric drop): 1.4657320715487003 ms # noqa: E501

model.eval().to(device)
dynamic_info = {
    "inputs": [
        {0: "batch", 1: "num_tokens"},
        {0: "batch", 1: "num_tokens"},
        {0: "batch", 1: "num_tokens"},
    ],
    "outputs": [{0: "batch", 1: "num_tokens"}],
}
speedster_optimized_model = optimize_model(
    model=model,
    input_data=encoded_inputs,
    optimization_time="constrained",
    # force it to use fastertransformer
    ignore_compilers=["tensor_rt", "tvm", "onnxruntime", "torchscript"],
    dynamic_info=dynamic_info,
)


benchmark(
    speedster_optimized_model, "speedster optimized BERT (no metric drop)"
)
benchmark(
    speedster_optimized_model, "speedster optimized BERT (no metric drop)"
)
# Average response time for speedster optimized BERT (no metric drop): 14.040142675396055 ms # noqa: E501
# Average response time for speedster optimized BERT (no metric drop): 3.4986357542220503 ms # noqa: E501
speedster_optimized_model_fp16 = optimize_model(
    model=model,
    input_data=encoded_inputs,
    optimization_time="constrained",
    # force it to use fastertransformer
    ignore_compilers=["tensor_rt", "tvm", "onnxruntime", "torchscript"],
    dynamic_info=dynamic_info,
    metric_drop_ths=0.1,
)


benchmark(
    speedster_optimized_model_fp16, "speedster optimized BERT (metric drop)"
)
benchmark(
    speedster_optimized_model_fp16, "speedster optimized BERT (metric drop)"
)
# Average response time for speedster optimized BERT (no metric drop): 14.040142675396055 ms # noqa: E501
# Average response time for speedster optimized BERT (no metric drop): 3.4986357542220503 ms # noqa: E501
