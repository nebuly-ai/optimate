import logging
import subprocess
import tempfile
import venv


class EnvBuilder(venv.EnvBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = None

    def post_setup(self, context):
        self.context = context


def run_in_different_venv(
    requirements_file: str,
    script_path: str,
    *args,
    logger: logging.Logger = None,
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
        logger (Logger, optional): Logger for the project.
    """
    if logger is not None:
        logger.debug(
            f"Debug: Running script {script_path} in a new virtual env."
        )
    with tempfile.TemporaryDirectory() as target_dir_path:
        if logger is not None:
            logger.debug("Debug: Creating virtual environment...")
        venv_builder = EnvBuilder(with_pip=True)
        venv_builder.create(str(target_dir_path))
        venv_context = venv_builder.context

        if logger is not None:
            logger.debug("Debug: Installing requirements...")
        pip_install_command = [
            venv_context.env_exe,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_file,
        ]
        subprocess.check_call(pip_install_command)

        if logger is not None:
            logger.debug("Debug: Executing script...")
        script_command = [venv_context.env_exe, script_path, *args]
        subprocess.check_call(script_command)
