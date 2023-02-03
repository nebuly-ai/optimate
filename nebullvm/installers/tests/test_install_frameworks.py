from nebullvm.installers.auto_installer import (
    select_frameworks_to_install,
    select_compilers_to_install,
)


def test_install_default_option():
    include_frameworks = "all"
    include_backends = "all"

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["huggingface", "onnx", "tensorflow", "torch"]


def test_install_torch_full():
    include_frameworks = ["torch"]
    include_backends = "all"

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["onnx", "torch"]


def test_install_torch_base():
    include_frameworks = ["torch"]
    include_backends = []

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["torch"]


def test_install_tensorflow_full():
    include_frameworks = ["tensorflow"]
    include_backends = "all"

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["onnx", "tensorflow"]


def test_install_tensorflow_base():
    include_frameworks = ["tensorflow"]
    include_backends = []

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["tensorflow"]


def test_install_onnx_full():
    include_frameworks = ["onnx"]
    include_backends = "all"

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["onnx"]


def test_install_onnx_base():
    include_frameworks = ["onnx"]
    include_backends = []

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["onnx"]


def test_install_huggingface_full():
    include_frameworks = ["huggingface"]
    include_backends = "all"

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["huggingface", "onnx", "tensorflow", "torch"]


def test_install_huggingface_full_tf():
    include_frameworks = ["huggingface"]
    include_backends = ["onnx", "tensorflow"]

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["huggingface", "onnx", "tensorflow"]


def test_install_huggingface_full_torch():
    include_frameworks = ["huggingface"]
    include_backends = ["onnx", "torch"]

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["huggingface", "onnx", "torch"]


def test_install_huggingface_tf():
    include_frameworks = ["huggingface"]
    include_backends = ["tensorflow"]

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["huggingface", "tensorflow"]


def test_install_huggingface_torch():
    include_frameworks = ["huggingface"]
    include_backends = ["torch"]

    include_backends = select_frameworks_to_install(
        include_frameworks, include_backends
    )

    assert include_backends == ["huggingface", "torch"]


def test_install_huggingface_compilers_all():
    framework_list = ["huggingface"]
    include_compilers = "all"

    compiler_list = select_compilers_to_install(
        include_compilers, framework_list
    )

    assert compiler_list == []


def test_install_huggingface_torch_compilers_all():
    framework_list = ["huggingface", "torch"]
    include_compilers = "all"

    compiler_list = select_compilers_to_install(
        include_compilers, framework_list
    )

    assert compiler_list == [
        "deepsparse",
        "intel_neural_compressor",
        "tensor_rt",
        "torch_tensor_rt",
    ]


def test_install_torch_compilers_all():
    framework_list = ["torch"]
    include_compilers = "all"

    compiler_list = select_compilers_to_install(
        include_compilers, framework_list
    )

    assert compiler_list == [
        "deepsparse",
        "intel_neural_compressor",
        "tensor_rt",
        "torch_tensor_rt",
    ]


def test_install_torch_compilers_deepsparse():
    framework_list = ["torch"]
    include_compilers = ["deepsparse"]

    compiler_list = select_compilers_to_install(
        include_compilers, framework_list
    )

    assert compiler_list == ["deepsparse"]


def test_install_torch_compilers_invalid():
    framework_list = ["torch"]
    include_compilers = ["best_compiler"]

    compiler_list = select_compilers_to_install(
        include_compilers, framework_list
    )

    assert compiler_list == []


def test_install_torch_onnx_compilers_all():
    framework_list = ["torch", "onnx"]
    include_compilers = "all"

    compiler_list = select_compilers_to_install(
        include_compilers, framework_list
    )

    assert compiler_list == [
        "deepsparse",
        "intel_neural_compressor",
        "openvino",
        "tensor_rt",
        "torch_tensor_rt",
    ]


def test_install_tensorflow_compilers_all():
    framework_list = ["tensorflow"]
    include_compilers = "all"

    compiler_list = select_compilers_to_install(
        include_compilers, framework_list
    )

    assert compiler_list == []
