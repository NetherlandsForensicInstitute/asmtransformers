from enum import StrEnum
from importlib import resources


class Architecture(StrEnum):
    AMD64 = 'amd64'
    ARM64 = 'arm64'
    I386 = 'i386'
    RISCV64 = 'riscv64'


def model_resource(name):
    return resources.files(__package__).joinpath(name)
