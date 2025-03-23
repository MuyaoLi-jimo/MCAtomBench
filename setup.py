import pathlib
import pkg_resources
from setuptools import setup, find_packages


def _read_file(fname):
    with pathlib.Path(fname).open(encoding="utf-8") as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]

setup(
    name="mcabench",
    version="0.0.1",
    author="craftjarvis",
    author_email="craftjarvis@outlook.com",
    long_description=_read_file("README.md"),
    description="A Benchmark to Test the Capability of Agent in Minecraft ",
    url="https://github.com/MuyaoLi-jimo/MCAtomBench",
    project_urls={
        "Bug Tracker": "https://github.com/MuyaoLi-jimo/MCAtomBench/issues",
    },
    install_requires=_read_install_requires(),
    packages=find_packages(),
    python_requires=">=3.9",
)