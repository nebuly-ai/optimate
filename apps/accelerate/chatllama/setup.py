from pathlib import Path
from setuptools import setup, find_packages


REQUIREMENTS = [
    "beartype",
    "deepspeed",
    "einops",
    "fairscale",
    "langchain>=0.0.103",
    "torch",
    "tqdm",
    "transformers",
    "datasets",
    "openai",
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf8")

setup(
    name="chatllama-py",
    version="0.0.2",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
)
