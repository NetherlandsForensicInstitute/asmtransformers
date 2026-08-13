from functools import cache
from importlib import resources


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


def default_model_kwargs(*, model_kwargs=None, device=None):
    import torch

    model_kwargs = model_kwargs or {}

    match device:
        case None:
            device = default_device()
        case str():
            device = torch.device(device)

    if device.type == 'xpu':
        # SPDA might cause runtime errors on xpu, default to eager attention to avoid it
        model_kwargs.setdefault('attn_implementation', 'eager')

    return model_kwargs
