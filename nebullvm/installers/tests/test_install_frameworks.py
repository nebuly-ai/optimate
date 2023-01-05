from nebullvm.installers.auto_installer import check_backends


def test_install_all_backends():
    include_backends = "all"

    include_backends = check_backends(include_backends)

    assert include_backends == ["huggingface", "onnx", "tensorflow", "torch"]


def test_install_torch_full():
    include_backends = ["torch-full"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["onnx", "torch"]


def test_install_torch_base():
    include_backends = ["torch-base"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["torch"]


def test_install_tensorflow_full():
    include_backends = ["tensorflow-full"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["onnx", "tensorflow"]


def test_install_tensorflow_base():
    include_backends = ["tensorflow-base"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["tensorflow"]


def test_install_onnx_full():
    include_backends = ["onnx-full"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["onnx"]


def test_install_onnx_base():
    include_backends = ["onnx-base"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["onnx"]


def test_install_huggingface_full():
    include_backends = ["huggingface-full"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["huggingface", "onnx", "tensorflow", "torch"]


def test_install_huggingface_full_tf():
    include_backends = ["huggingface-full-tf"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["huggingface", "onnx", "tensorflow"]


def test_install_huggingface_full_torch():
    include_backends = ["huggingface-full-torch"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["huggingface", "onnx", "torch"]


def test_install_huggingface_tf():
    include_backends = ["huggingface-base-tf"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["huggingface", "tensorflow"]


def test_install_huggingface_torch():
    include_backends = ["huggingface-base-torch"]

    include_backends = check_backends(include_backends)

    assert include_backends == ["huggingface", "torch"]
