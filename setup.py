from pathlib import Path
from setuptools import setup, find_packages


REQUIREMENTS = [
    "numpy>=1.20.0",
    "py-cpuinfo>=8.0.0",
    "PyYAML>=6.0",
    "psutil>=5.0.0",
    "requests>=2.26.0",
    "tqdm>=4.36.0",
    "packaging>=21.3",
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nebullvm",
    version="0.4.4",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
)
