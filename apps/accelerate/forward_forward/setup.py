from pathlib import Path
from setuptools import setup, find_packages


REQUIREMENTS = [
    "torch>=1.9",
    "torchvision>=0.10",
    "nebullvm>=0.6",
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf8")

setup(
    name="forward_forward",
    version="0.0.1",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
)
