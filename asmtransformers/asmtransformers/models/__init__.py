from contextlib import suppress
from functools import cache
from importlib import import_module, resources


def model_resource(name):
    return resources.files(__package__).joinpath(name)


@cache
def default_device():
    import torch

    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.xpu.is_available():
        return torch.device('xpu')
    if torch.mps.is_available():
        return torch.device('mps')

    # fall back to using cpu
    return torch.device('cpu')
