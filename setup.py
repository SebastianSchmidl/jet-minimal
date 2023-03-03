from typing import List
from setuptools import setup, find_packages


def load_dependencies() -> List[str]:
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="jet",
    description="JET: Jaunty Estimation of Hierarchical Time Series Clustering",
    version="0.1.0",
    packages=find_packages(),
    install_requires=load_dependencies(),
    python_requires=">=3.9",
)
