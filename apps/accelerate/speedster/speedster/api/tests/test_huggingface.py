from tempfile import TemporaryDirectory

from nebullvm.config import COMPILER_LIST, COMPRESSOR_LIST
from nebullvm.operations.inference_learners.huggingface import (
    HuggingFaceInferenceLearner,
)
from nebullvm.optional_modules.tensorflow import tensorflow as tf
from nebullvm.optional_modules.torch import torch
from transformers import AlbertModel, TFAlbertModel, AlbertTokenizer

from speedster import optimize_model, load_model


def test_torch_huggingface_ort_input_text():
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    model = AlbertModel.from_pretrained("albert-base-v1")

    # Move the model to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_data = [
        "this is a test",
        "hi my name is Valerio",
        "india is very far from italy",
    ]

    optimized_model = optimize_model(
        model=model,
        input_data=input_data,
        optimization_time="constrained",
        tokenizer=tokenizer,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "onnxruntime"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        tokenizer_args=dict(
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
            return_token_type_ids=None,  # Sets to model default
            padding="longest",
            truncation=True,
        ),
    )

    # save and load
    with TemporaryDirectory() as tmp_dir:
        optimized_model.save(tmp_dir)
        loaded_model = load_model(tmp_dir)
        assert isinstance(loaded_model, HuggingFaceInferenceLearner)

        assert isinstance(loaded_model.get_size(), int)

    x = ["this is a test input to see if the optimized model works."]
    inputs = tokenizer(x, return_tensors="pt").to(device)
    model.to(device)
    res_original = model(**inputs)
    res_optimized = optimized_model(**inputs)

    assert isinstance(optimized_model, HuggingFaceInferenceLearner)

    assert (
        torch.max(
            abs(
                (
                    res_original["last_hidden_state"]
                    - res_optimized["last_hidden_state"]
                )
            )
        )
        < 1e-2
    )
    assert (
        torch.max(
            abs(
                (
                    res_original["pooler_output"]
                    - res_optimized["pooler_output"]
                )
            )
        )
        < 1e-2
    )


def test_torch_huggingface_ort_input_tensors():
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    model = AlbertModel.from_pretrained("albert-base-v1")

    # Move the model to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text = "hi my name is Valerio"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    dynamic_info = {
        "inputs": [
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
        ],
        "outputs": [{0: "batch", 1: "num_tokens"}, {0: "batch"}],
    }

    optimized_model = optimize_model(
        model=model,
        input_data=[inputs for _ in range(10)],
        optimization_time="constrained",
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "onnxruntime"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        dynamic_info=dynamic_info,
    )

    x = ["this is a test input to see if the optimized model works."]
    inputs = tokenizer(x, return_tensors="pt").to(device)
    model.to(device)
    res_original = model(**inputs)
    res_optimized = optimized_model(**inputs)

    assert isinstance(optimized_model, HuggingFaceInferenceLearner)

    assert (
        torch.max(
            abs(
                (
                    res_original["last_hidden_state"]
                    - res_optimized["last_hidden_state"]
                )
            )
        )
        < 1e-2
    )
    assert (
        torch.max(
            abs(
                (
                    res_original["pooler_output"]
                    - res_optimized["pooler_output"]
                )
            )
        )
        < 1e-2
    )


def test_torch_huggingface_torchscript_input_tensors():
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    model = AlbertModel.from_pretrained("albert-base-v1", torchscript=True)

    # Move the model to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text = "hi my name is Valerio"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    dynamic_info = {
        "inputs": [
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
        ],
        "outputs": [{0: "batch", 1: "num_tokens"}, {0: "batch"}],
    }

    optimized_model = optimize_model(
        model=model,
        input_data=[inputs for _ in range(10)],
        optimization_time="constrained",
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "torchscript"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        dynamic_info=dynamic_info,
    )

    x = ["this is a test input to see if the optimized model works."]
    inputs = tokenizer(x, return_tensors="pt").to(device)
    model.to(device)
    res_original = model(**inputs)
    res_optimized = optimized_model(**inputs)

    assert isinstance(optimized_model, HuggingFaceInferenceLearner)

    assert torch.max(abs((res_original[0] - res_optimized[0]))) < 1e-2
    assert torch.max(abs((res_original[1] - res_optimized[1]))) < 1e-2


def test_tensorflow_huggingface_ort_input_text_np():
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    model = TFAlbertModel.from_pretrained("albert-base-v1")

    input_data = [
        "this is a test",
        "hi my name is Valerio",
        "india is very far from italy",
    ]

    dynamic_info = {
        "inputs": [
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
        ],
        "outputs": [{0: "batch", 1: "num_tokens"}, {0: "batch"}],
    }

    optimized_model = optimize_model(
        model=model,
        input_data=input_data,
        optimization_time="constrained",
        tokenizer=tokenizer,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "onnxruntime"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        tokenizer_args=dict(
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="np",
            return_token_type_ids=None,  # Sets to model default
            padding="longest",
            truncation=True,
        ),
        dynamic_info=dynamic_info,
    )

    x = ["this is a test input to see if the optimized model works."]
    inputs = tokenizer(x, return_tensors="np")
    res_original = model(**inputs)
    res_optimized = optimized_model(**inputs)

    assert isinstance(optimized_model, HuggingFaceInferenceLearner)

    assert (
        tf.math.reduce_max(
            abs(
                (
                    res_original["last_hidden_state"]
                    - res_optimized["last_hidden_state"]
                )
            )
        )
        < 1e-2
    )
    assert (
        tf.math.reduce_max(
            abs(
                (
                    res_original["pooler_output"]
                    - res_optimized["pooler_output"]
                )
            )
        )
        < 1e-2
    )


def test_tensorflow_huggingface_ort_input_tensors_np():
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    model = TFAlbertModel.from_pretrained("albert-base-v1")

    text = "hi my name is Valerio"
    inputs = tokenizer(text, return_tensors="np")

    dynamic_info = {
        "inputs": [
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
        ],
        "outputs": [{0: "batch", 1: "num_tokens"}, {0: "batch"}],
    }

    optimized_model = optimize_model(
        model=model,
        input_data=[inputs for _ in range(10)],
        optimization_time="constrained",
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "onnxruntime"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        dynamic_info=dynamic_info,
    )

    x = ["Test to see if it works with a different output"]
    inputs = tokenizer(x, return_tensors="np")
    res_original = model(**inputs)
    res_optimized = optimized_model(**inputs)

    assert isinstance(optimized_model, HuggingFaceInferenceLearner)

    assert (
        tf.math.reduce_max(
            abs(
                (
                    res_original["last_hidden_state"]
                    - res_optimized["last_hidden_state"]
                )
            )
        )
        < 1e-2
    )
    assert (
        tf.math.reduce_max(
            abs(
                (
                    res_original["pooler_output"]
                    - res_optimized["pooler_output"]
                )
            )
        )
        < 1e-2
    )


def test_tensorflow_huggingface_ort_input_text_tf():
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    model = TFAlbertModel.from_pretrained("albert-base-v1")

    input_data = [
        "this is a test",
        "hi my name is Valerio",
        "india is very far from italy",
    ]

    dynamic_info = {
        "inputs": [
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
        ],
        "outputs": [{0: "batch", 1: "num_tokens"}, {0: "batch"}],
    }

    optimized_model = optimize_model(
        model=model,
        input_data=input_data,
        optimization_time="constrained",
        tokenizer=tokenizer,
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "onnxruntime"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        tokenizer_args=dict(
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="tf",
            return_token_type_ids=None,  # Sets to model default
            padding="longest",
            truncation=True,
        ),
        dynamic_info=dynamic_info,
    )

    x = ["this is a test input to see if the optimized model works."]
    inputs = tokenizer(x, return_tensors="tf")
    res_original = model(**inputs)
    res_optimized = optimized_model(**inputs)

    assert isinstance(optimized_model, HuggingFaceInferenceLearner)

    assert (
        tf.math.reduce_max(
            abs(
                (
                    res_original["last_hidden_state"]
                    - res_optimized["last_hidden_state"]
                )
            )
        )
        < 1e-2
    )
    assert (
        tf.math.reduce_max(
            abs(
                (
                    res_original["pooler_output"]
                    - res_optimized["pooler_output"]
                )
            )
        )
        < 1e-2
    )


def test_tensorflow_huggingface_ort_input_tensors_tf():
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
    model = TFAlbertModel.from_pretrained("albert-base-v1")

    text = "hi my name is Valerio"
    inputs = tokenizer(text, return_tensors="tf")

    dynamic_info = {
        "inputs": [
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
            {0: "batch", 1: "num_tokens"},
        ],
        "outputs": [{0: "batch", 1: "num_tokens"}, {0: "batch"}],
    }

    optimized_model = optimize_model(
        model=model,
        input_data=[inputs for _ in range(10)],
        optimization_time="constrained",
        ignore_compilers=[
            compiler for compiler in COMPILER_LIST if compiler != "onnxruntime"
        ],
        ignore_compressors=[compressor for compressor in COMPRESSOR_LIST],
        dynamic_info=dynamic_info,
    )

    x = ["Test to see if it works with a different output"]
    inputs = tokenizer(x, return_tensors="tf")
    res_original = model(**inputs)
    res_optimized = optimized_model(**inputs)

    assert isinstance(optimized_model, HuggingFaceInferenceLearner)

    assert (
        tf.math.reduce_max(
            abs(
                (
                    res_original["last_hidden_state"]
                    - res_optimized["last_hidden_state"]
                )
            )
        )
        < 1e-2
    )
    assert (
        tf.math.reduce_max(
            abs(
                (
                    res_original["pooler_output"]
                    - res_optimized["pooler_output"]
                )
            )
        )
        < 1e-2
    )
