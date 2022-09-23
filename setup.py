# import platform
from pathlib import Path
from setuptools import setup, find_packages


REQUIREMENTS = [
    "numpy>=1.20.0, <1.23.0",
    "onnx>=1.10.0",
    "onnxmltools>=1.11.0",
    "py-cpuinfo>=8.0.0",
    "PyYAML>=6.0",
    "psutil>=5.9.0",
    "requests>=2.28.1",
    "torch>=1.10.0",
    "tqdm>=4.63.0",
    "packaging>=21.3",
]
#
# if platform.system() != "Darwin":
#     REQUIREMENTS.append("scipy==1.5.4")

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nebullvm",
    version="0.4.3",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    include_package_data=True,
    long_description_content_type="text/markdown",
)
