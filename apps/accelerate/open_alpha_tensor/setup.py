from pathlib import Path
from setuptools import setup, find_packages


REQUIREMENTS = [
    "nebullvm",
    "torch",
    "tqdm",
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf8")

setup(
    name="OpenAlphaTensor",
    version="0.0.1",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
)
