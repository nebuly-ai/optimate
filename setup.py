from setuptools import setup, find_packages

setup(
    name="nebullvm",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        # Include all bash files:
        "": ["*.sh"],
        # And include any file needed for config
        "nebullvm": ["*config.cmake"],
    },
)
