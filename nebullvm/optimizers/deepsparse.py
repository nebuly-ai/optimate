from imageio import save
from pathlib import Path
from sparseml.onnx.optim import ModelAnalyzer
from sparsify.schemas import ProjectModelAnalysisSchema
from sparseml.onnx.optim import pruning_loss_sens_magnitude
from sparsify.blueprints.utils import PruningModelEvaluator
from sparsify.blueprints.utils import default_pruning_settings
from sparseml.pytorch.sparsification import (
            EpochRangeModifier,
            GMPruningModifier)
from tempfile import TemporaryDirectory
from sparsify.blueprints.utils import default_epochs_distribution
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim import (
    ScheduledModifierManager,
)
from sparseml.pytorch.utils import ModuleExporter
from tqdm.auto import tqdm
from nebullvm.utils.data import DataManager
from typing import Optional, Callable
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from logging import Logger
from nebullvm.base import SparsityParams, ModelParams, DeepLearningFramework
from nebullvm.transformations.base import MultiStageTransformation
from nebullvm.inference_learners.deepsparse import DEEPSPARSE_INFERENCE_LEARNERS, DeepSparseInferenceLearner
from nebullvm.optimizers.quantization.utils import (
    check_precision
)

from nebullvm.utils.onnx import (
    get_input_names,
    get_output_names,
    run_onnx_model,
    convert_to_target_framework,
)


def _export_model_onnx(model, save_path, model_name, input_infos, bs=1):
    exporter = ModuleExporter(model, output_dir=save_path)
    # TODO: handle case when model has more than one input
    input_shape = (bs, *input_infos[0].size)
    exporter.export_onnx(torch.randn(input_shape), name=model_name)
    onnx_path = save_path / model_name

    return onnx_path


class DeepSparseOptimizer():
    def __init__(self, logger: Logger = None):
        self.logger = logger

    def optimize_from_torch(
        self,
        torch_model: torch.nn.Module,
        model_params: ModelParams = None,
        sparsity_params: SparsityParams = None,
        input_tfms: MultiStageTransformation = None,
        perf_metric: Callable = None,
        perf_loss_ths: float = None,
        input_data: DataManager = None
    ) -> Optional[DeepSparseInferenceLearner]:
        batch_size = sparsity_params.finetuning_batch_size
        train_data_loader = sparsity_params.train_dataloader
        val_data_loader = sparsity_params.val_dataloader
        output_library = DeepLearningFramework.PYTORCH


        with TemporaryDirectory() as tmp_dir:
            onnx_path = _export_model_onnx(torch_model, Path(tmp_dir), "model.onnx", model_params.input_infos)
            onnx_path = onnx_path.as_posix()

            input_data_onnx, output_data_onnx, ys = [], [], None
            #input_data = train_data_loader
            if perf_loss_ths is not None:
                input_data_onnx, ys = input_data.get_numpy_list(
                        300, with_ys=True
                    )
                output_data_onnx = [
                    tuple(run_onnx_model(onnx_path, list(input_tensors)))
                    for input_tensors in input_data_onnx
                ]


            recipe = RecipeBuilder(onnx_path)
            # TODO: implement custom parameters support
            manager = recipe.build_recipe()
            trainer = PruningTrainer(torch_model, batch_size)
            pruned_model = trainer.train(manager, train_data_loader, val_data_loader)
            onnx_pruned_path = _export_model_onnx(pruned_model, Path(tmp_dir), "model_pruned.onnx", model_params.input_infos)
            onnx_pruned_path = onnx_pruned_path.as_posix()

            
            learner = DEEPSPARSE_INFERENCE_LEARNERS[output_library](
                input_tfms=input_tfms,
                network_parameters=model_params,
                onnx_path=onnx_pruned_path,
                input_names=get_input_names(onnx_pruned_path),
                output_names=get_output_names(onnx_pruned_path),
            )

            if perf_loss_ths is not None:
                inputs = [
                    tuple(
                        convert_to_target_framework(t, output_library)
                        for t in data_tuple
                    )
                    for data_tuple in input_data_onnx
                ]
                is_valid = check_precision(
                    learner,
                    inputs,
                    output_data_onnx,
                    perf_loss_ths,
                    metric_func=perf_metric,
                    ys=ys,
                )
                if not is_valid:
                    return None
        
        return learner


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
            "baseline_measurement_key": (str(results_model.baseline_measurement_key)),
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
                    "baseline_measurement_key": (str(res.baseline_measurement_key)),
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

    def build_recipe(self, custom_parameters=None, training_epochs=10):
        self._make_analysis()
        self._compute_loss_sensitivity()

        if custom_parameters is None:
            epochs = default_epochs_distribution(training_epochs)
        else:
            # TODO: set custom parameters
            epochs = None
            pass

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

    def _setup_training(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.criterion = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def _run_model_one_epoch(self, train=False):

        if train:
            self.model.train()
            data_loader = self.train_data_loader
        else:
            self.model.eval()
            data_loader = self.val_data_loader

        running_loss = 0.0
        total_correct = 0
        total_predictions = 0

        for step, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if train:
                self.optimizer.zero_grad()

            outputs, _ = self.model(inputs)  # model returns logits and softmax as a tuple
            loss = self.criterion(outputs, labels)

            if train:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            total_correct += torch.sum(predictions == labels).item()
            total_predictions += inputs.size(0)

        loss = running_loss / (step + 1.0)
        accuracy = total_correct / total_predictions
        return loss, accuracy

    def train(self, manager, train_data_loader, val_data_loader):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self._setup_training()
        self.optimizer = manager.modify(self.model, self.optimizer, steps_per_epoch=len(self.train_data_loader))
        self.model.train()
        # Run model pruning
        epoch = manager.min_epochs
        while epoch < manager.max_epochs:
            # run training loop
            epoch_name = "{}/{}".format(epoch + 1, manager.max_epochs)
            print("Running Training Epoch {}".format(epoch_name))
            train_loss, train_acc = self._run_model_one_epoch(train=True)
            print(
                "Training Epoch: {}\nTraining Loss: {}\nTop 1 Acc: {}\n".format(
                    epoch_name, train_loss, train_acc
                )
            )

            # run validation loop
            print("Running Validation Epoch {}".format(epoch_name))
            val_loss, val_acc = self._run_model_one_epoch()
            print(
                "Validation Epoch: {}\nVal Loss: {}\nTop 1 Acc: {}\n".format(
                    epoch_name, val_loss, val_acc
                )
            )

            epoch += 1

        manager.finalize(self.model)

        return self.model
