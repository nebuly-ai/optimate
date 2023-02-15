from pathlib import Path
from setuptools import setup, find_packages


REQUIREMENTS = [
    "nebullvm>=0.8.1",
    "tabulate>=0.8.0",
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf8")

setup(
    name="speedster",
    version="0.2.1",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
)
