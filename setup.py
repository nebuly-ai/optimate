from setuptools import setup, find_packages

REQUIREMENTS = [
    "numpy>=1.19.0",
    "joblib>=1.1.0",
    "onnx>=1.10.0",
    "py-cpuinfo>=8.0.0",
    "tensorflow~=2.7.0",
    "tf2onnx~=1.8.4",
    "torch>=1.10.0",
]

setup(
    name="nebullvm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    package_data={
        # Include all bash files:
        "": ["*.sh"],
        # And include any file needed for config
        "nebullvm": ["*config.cmake"],
    },
)
