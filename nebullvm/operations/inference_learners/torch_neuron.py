import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nebullvm.operations.inference_learners.torchscript import (
    TorchScriptInferenceLearner,
)


class TorchNeuronInferenceLearner(TorchScriptInferenceLearner):
    name = "TorchNeuron"

    def get_size(self):
        with TemporaryDirectory() as tmp_dir:
            self.save(tmp_dir)
            return sum(
                os.path.getsize(Path(tmp_dir) / f)
                for f in os.listdir(Path(tmp_dir))
                if os.path.isfile(Path(tmp_dir) / f)
            )
