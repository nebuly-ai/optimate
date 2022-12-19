from torchvision import datasets

from forward_forward.root_op import (
    ForwardForwardRootOp,
    ForwardForwardModelType,
)


def train_with_forward_forward_algorithm(
    n_layers: int = 2,
    model_type: str = "progressive",
    device: str = "cpu",
    hidden_size: int = 2000,
    lr: float = 0.03,
    epochs: int = 100,
    batch_size: int = 5000,
    theta: float = 2.0,
    shuffle: bool = True,
    **kwargs,
):
    model_type = ForwardForwardModelType(model_type)
    root_op = ForwardForwardRootOp(model_type)

    output_size = None
    if model_type is ForwardForwardModelType.PROGRESSIVE:
        input_size = 28 * 28 + len(datasets.MNIST.classes)
    elif model_type is ForwardForwardModelType.RECURRENT:
        input_size = 28 * 28
        output_size = len(datasets.MNIST.classes)
    else:  # model_type is ForwardForwardModelType.NLP
        input_size = 10  # number of characters
        output_size = 30  # length of vocabulary
        assert (
            kwargs.get("predicted_tokens") is not None
        ), "predicted_tokens must be specified for NLP model"

    root_op.execute(
        input_size=input_size,
        n_layers=n_layers,
        hidden_size=hidden_size,
        optimizer_name="Adam",
        optimizer_params={"lr": lr},
        loss_fn_name="alternative_loss_fn",
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        shuffle=shuffle,
        theta=theta,
        output_size=output_size,
    )

    return root_op.get_result()
