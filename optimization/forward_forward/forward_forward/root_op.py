from enum import Enum

from nebullvm.operations.base import Operation

from forward_forward.operations.build_models import (
    FCNetFFProgressiveBuildOperation,
    RecurrentFCNetFFBuildOperation,
    LMFFNetBuildOperation,
)
from forward_forward.operations.data import (
    MNISTDataLoaderOperation,
    AesopFablesDataLoaderOperation,
)
from forward_forward.operations.trainers import (
    ForwardForwardTrainer,
    RecurrentForwardForwardTrainer,
    NLPForwardForwardTrainer,
)


class ForwardForwardModelType(Enum):
    PROGRESSIVE = "progressive"
    RECURRENT = "recurrent"
    NLP = "nlp"


class ForwardForwardRootOp(Operation):
    def __init__(self, model_type: ForwardForwardModelType):
        super().__init__()

        if model_type is ForwardForwardModelType.PROGRESSIVE:
            self.build_model = FCNetFFProgressiveBuildOperation()
            self.train_model = ForwardForwardTrainer()
            self.load_data = MNISTDataLoaderOperation()
        elif model_type is ForwardForwardModelType.RECURRENT:
            self.build_model = RecurrentFCNetFFBuildOperation()
            self.train_model = RecurrentForwardForwardTrainer()
            self.load_data = MNISTDataLoaderOperation()
        elif model_type is ForwardForwardModelType.NLP:
            self.build_model = LMFFNetBuildOperation()
            self.train_model = NLPForwardForwardTrainer()
            self.load_data = AesopFablesDataLoaderOperation()

    def execute(
        self,
        input_size: int,
        n_layers: int,
        hidden_size: int,
        optimizer_name: str,
        optimizer_params: dict,
        loss_fn_name: str,
        batch_size: int,
        epochs: int,
        shuffle: bool,
        theta: float,
        device: str,
        output_size: int = None,
        **kwargs,
    ):
        if self.build_model.get_result() is None:
            self.build_model.execute(
                input_size=input_size,
                n_layers=n_layers,
                hidden_size=hidden_size,
                optimizer_name=optimizer_name,
                optimizer_params=optimizer_params,
                loss_fn_name=loss_fn_name,
                output_size=output_size,
            )

        if self.load_data.get_result() is None:
            self.load_data.execute(batch_size=batch_size, shuffle=shuffle)

        if (
            self.build_model.get_result() is not None
            and self.load_data.get_result() is not None
        ):
            if self.train_model.get_result() is None:
                train_loader, test_loader = self.load_data.get_result()
                self.train_model.execute(
                    model=self.build_model.get_result(),
                    train_data=train_loader,
                    test_data=test_loader,
                    epochs=epochs,
                    theta=theta,
                    device=device,
                    **kwargs,
                )
            if self.train_model.get_result() is not None:
                self.state["model"] = self.train_model.get_result()

    def get_result(self):
        return self.state.get("model")
