import subprocess
import tempfile
import venv

from loguru import logger


class EnvBuilder(venv.EnvBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = None

    def post_setup(self, context):
        self.context = context


def run_in_different_venv(
    requirements_file: str,
    script_path: str,
    use_gpu: bool,
    *args,
):
    """Run a python scripts in a new temporary environment. Arguments for the
    script must be passed in the function args.
    it is equivalent to create and activate a new environment and running
    > pip install -r $requirement_file
    > python -m script_path *args
    Args:
        requirements_file (str): File (.txt) containing the list of
            requirements.
        script_path (str): Path to the script that must be run.
        args: Arguments of the script.
    """
    logger.debug(f"Debug: Running script {script_path} in a new virtual env.")
    with tempfile.TemporaryDirectory() as target_dir_path:
        logger.debug("Debug: Creating virtual environment...")
        venv_builder = EnvBuilder(with_pip=True)
        venv_builder.create(str(target_dir_path))
        venv_context = venv_builder.context

        logger.debug("Debug: Installing requirements...")

        if use_gpu:
            pip_install_command = [
                venv_context.env_exe,
                "-m",
                "pip",
                "install",
                "torch==1.9.1+cu111",
                "torchvision==0.10.1+cu111",
                "-f",
                "https://download.pytorch.org/whl/torch_stable.html",
            ]
        else:
            pip_install_command = [
                venv_context.env_exe,
                "-m",
                "pip",
                "install",
                "torch<=1.9.1",
                "torchvision<=0.10.1",
            ]
        subprocess.check_call(pip_install_command)

        pip_install_command = [
            venv_context.env_exe,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_file,
        ]
        subprocess.check_call(pip_install_command)

        logger.debug("Debug: Executing script...")
        script_command = [venv_context.env_exe, script_path, *args]
        subprocess.check_call(script_command)
