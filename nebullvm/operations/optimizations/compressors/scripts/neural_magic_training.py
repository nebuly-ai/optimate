import json
import logging
import os.path
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, List, Any, Dict

import torch
from sparseml.onnx.optim import ModelAnalyzer, pruning_loss_sens_magnitude
from sparseml.pytorch.optim import (
    ScheduledModifierManager,
)
from sparseml.pytorch.sparsification import (
    EpochRangeModifier,
    GMPruningModifier,
)
from sparseml.pytorch.utils import ModuleExporter
from sparsify.blueprints.utils import (
    default_epochs_distribution,
    PruningModelEvaluator,
    default_pruning_settings,
)
from sparsify.schemas import ProjectModelAnalysisSchema
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD
from tqdm.auto import tqdm

CRITERION_FNS = {
    "CrossEntropy": CrossEntropyLoss(),
    "MSE": MSELoss(),
}

logging.basicConfig(
    format=" %(asctime)s [%(levelname)s] %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S %p",
)
logger = logging.getLogger("nebullvm_logger")
logger.setLevel(logging.INFO)


def _export_model_onnx(
    model: torch.nn.Module,
    save_path: Path,
    model_name: str,
    input_batch: Tuple,
):
    if torch.cuda.is_available():
        input_batch = tuple(t.cuda() for t in input_batch)
        model.cuda()

    exporter = ModuleExporter(model, output_dir=save_path)
    with torch.no_grad():
        example_outputs = model(*input_batch)
    exporter.export_onnx(
        input_batch, name=model_name, example_outputs=example_outputs
    )
    onnx_path = save_path / model_name

    return onnx_path


class RecipeBuilder:
    def __init__(self, model_path):
        self.model_path = model_path

    def _make_analysis(self):
        analyzer = ModelAnalyzer(self.model_path)
        self.analysis = ProjectModelAnalysisSchema().load(analyzer.dict())

    def _compute_loss_sensitivity(self):
        sensitivities = []
        parameters = []
        for i, node in enumerate(self.analysis["nodes"]):
            if node["prunable"]:
                sensitivities.append(node["prunable_equation_sensitivity"])
                parameters.append(node["prunable_params"])

        loss_analysis = pruning_loss_sens_magnitude(self.model_path)

        results_model = loss_analysis.results_model
        results = loss_analysis.results

        model = {
            "baseline_measurement_key": (
                str(results_model.baseline_measurement_key)
            ),
            "measurements": {
                str(key): val for key, val in results_model.averages.items()
            },
        }
        ops = []

        for res in results:
            ops.append(
                {
                    "id": res.id_,
                    "name": res.name,
                    "index": res.index,
                    "baseline_measurement_key": (
                        str(res.baseline_measurement_key)
                    ),
                    "measurements": {
                        str(key): val for key, val in res.averages.items()
                    },
                }
            )

        pruning = {"model": model, "ops": ops}
        loss = {}
        loss["baseline"] = {}
        loss["pruning"] = pruning

        model = PruningModelEvaluator(
            self.analysis,
            None,
            loss,
        )
        model.eval_baseline(default_pruning_settings().sparsity)
        model.eval_pruning(default_pruning_settings())

        self.final_analysis = model.to_dict_values()

    def build_recipe(self, epochs_pruning_window=None, training_epochs=10):
        self._make_analysis()
        self._compute_loss_sensitivity()

        if epochs_pruning_window is None:
            epochs = default_epochs_distribution(training_epochs)
        else:
            # TODO: set custom parameters
            epochs = default_epochs_distribution(training_epochs)
            epochs_dict = epochs._asdict()
            epochs_dict.update(epochs_pruning_window)
            epochs = epochs.__class__(**epochs_dict)

        mods = [
            EpochRangeModifier(
                start_epoch=epochs.start_epoch,
                end_epoch=epochs.end_epoch,
            )
        ]

        node_weight_name_lookup = {
            node["id"]: node["weight_name"]
            for node in self.analysis["nodes"]
            if node["prunable"]
        }

        sparsity_to_params = {}

        nodes = self.final_analysis[0]

        for node in nodes:
            sparsity = node["sparsity"]
            node_id = node["node_id"]
            weight_name = node_weight_name_lookup[node_id]

            if sparsity is None:
                continue

            if sparsity not in sparsity_to_params:
                sparsity_to_params[sparsity] = []

            sparsity_to_params[sparsity].append(weight_name)

        for sparsity, params in sparsity_to_params.items():
            gm_pruning = GMPruningModifier(
                init_sparsity=0.05,
                final_sparsity=sparsity,
                start_epoch=epochs.pruning_start_epoch,
                end_epoch=epochs.pruning_end_epoch,
                update_frequency=epochs.pruning_update_frequency,
                params=params,
            )

            mods.append(gm_pruning)

        return ScheduledModifierManager(mods)


class PruningTrainer:
    def __init__(self, model, bs):
        self.data_loader = None
        self.optimizer = None
        self.model = model
        self.batch_size = bs

    def _setup_training(self, loss_fn=None, lr=1e-3, momentum=0.9):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if loss_fn is None:
            loss_fn = CrossEntropyLoss()
        else:
            loss_fn = CRITERION_FNS.get(loss_fn, CrossEntropyLoss())
        self.criterion = loss_fn
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum)

    def _run_model_one_epoch(self, train=False):

        if train:
            self.model.train()
            data_loader = self.train_data_loader
        else:
            self.model.eval()
            data_loader = self.val_data_loader

        running_loss = 0.0

        for step, (inputs, labels) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            inputs = tuple(t.to(self.device) for t in inputs)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
                if len(labels.shape) == 0:
                    labels = labels.unsqueeze(0)
            labels = labels.to(self.device)

            if train:
                self.optimizer.zero_grad()

            outputs = self.model(
                *inputs
            )  # model returns logits and softmax as a tuple
            loss = self.criterion(outputs, labels)

            if train:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        loss = running_loss / (len(data_loader) + 1e-5)
        return loss

    def train(
        self, manager, train_data_loader, val_data_loader, **train_kwargs
    ):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self._setup_training(**train_kwargs)
        self.optimizer = manager.modify(
            self.model,
            self.optimizer,
            steps_per_epoch=len(self.train_data_loader),
        )
        self.model.train()
        # Run model pruning
        epoch = manager.min_epochs
        while epoch < manager.max_epochs:
            # run training loop
            epoch_name = "{}/{}".format(epoch + 1, manager.max_epochs)
            logger.info("Running Training Epoch {}".format(epoch_name))
            train_loss = self._run_model_one_epoch(train=True)
            logger.info(
                ("Training Epoch: {}\nTraining Loss: {}\n").format(
                    epoch_name, train_loss
                )
            )

            # run validation loop
            logger.info("Running Validation Epoch {}".format(epoch_name))
            val_loss = self._run_model_one_epoch()
            logger.info(
                "Validation Epoch: {}\nVal Loss: {}\n".format(
                    epoch_name, val_loss
                )
            )

            epoch += 1

        manager.finalize(self.model)

        return self.model


def _load_config(config_file: str):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def _load_data(data_dir: str):
    data_dir = Path(data_dir)
    return [torch.load(input_path) for input_path in data_dir.glob("*.pt")]


def _load_model(model_file: str):
    if os.path.isdir(model_file):
        path = Path(model_file)
        module_file = path / "module.py"
        with open(module_file, "r") as f:
            module_str = f.read()
        exec(module_str, globals())
        model = eval("NebullvmFxModule")()
        model.load_state_dict(torch.load(path / "state_dict.pt"))
    else:
        model = torch.load(model_file)
    return model


def _train_model(
    model: torch.nn.Module,
    train_data: List[Tuple[Tuple, Any]],
    eval_data: List[Tuple[Tuple, Any]],
    epochs_pruning_window: Dict = None,
    training_epochs: int = 10,
    lr: float = 1e-3,
    momentum: float = 0.9,
    loss_fn: str = "CrossEntropy",
):
    batch_size = train_data[0][0][0].shape[0]
    with TemporaryDirectory() as tmp_dir:
        onnx_path = _export_model_onnx(
            model, Path(tmp_dir), "model.onnx", train_data[0][0]
        )
        onnx_path = onnx_path.as_posix()

        recipe = RecipeBuilder(onnx_path)
        # TODO: implement custom parameters support
        manager = recipe.build_recipe(
            epochs_pruning_window=epochs_pruning_window,
            training_epochs=training_epochs,
        )
        trainer = PruningTrainer(model, batch_size)
        pruned_model = trainer.train(
            manager, train_data, eval_data, lr=lr, momentum=momentum
        )
        return pruned_model


def _save_model(model: torch.nn.Module, path: str):
    if path.endswith(".pt"):
        torch.save(model, path)
    else:
        torch.save(model.state_dict(), Path(path) / "pruned_state_dict.pt")


def main(
    model_file: str,
    train_data_dir: str,
    eval_data_dir: str,
    config_file: str,
    out_file: str,
):
    config = _load_config(config_file)
    model = _load_model(model_file)
    train_data = _load_data(train_data_dir)
    eval_data = _load_data(eval_data_dir)
    pruned_model = _train_model(model, train_data, eval_data, **config)
    _save_model(pruned_model, out_file)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", help="The model to be pruned.")
    parser.add_argument(
        "--train_dir",
        help="The directory contained the pickled training data.",
    )
    parser.add_argument(
        "--eval_dir", help="The directory contained the pickled test data."
    )
    parser.add_argument("--config", help="The config file.")
    parser.add_argument(
        "--pruned_model", help="Path where storing the pruned model."
    )
    args = parser.parse_args()
    main(
        model_file=args.model,
        train_data_dir=args.train_dir,
        eval_data_dir=args.eval_dir,
        config_file=args.config,
        out_file=args.pruned_model,
    )
