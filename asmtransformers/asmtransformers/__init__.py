from enum import StrEnum


class Architecture(StrEnum):
    """
    An enumeration of the architectures supported by asmtransformers.
    """

    AMD64 = 'amd64'
    ARM64 = 'arm64'
    I386 = 'i386'
    RISCV64 = 'riscv64'
