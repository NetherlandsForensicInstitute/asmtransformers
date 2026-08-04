from importlib import resources


def model_resource(name):
    return resources.files(__package__).joinpath(name)
