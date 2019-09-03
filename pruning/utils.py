"""Auxiliary utils for implementing pruning strategies
"""

from ..metrics import model_size
from ..models.head import get_classifier_module


def get_modules(module, prefix=""):
    """Recursively find all submodules from torch modules,
    returning them in state_dict format
    """
    modules = {}
    for n, m in module.named_children():
        modules[prefix+n] = m
        modules.update(get_modules(m, prefix=prefix+n+'.'))
    return modules


def get_params(model, prefix=None):
    params = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    if prefix is not None:
        params = {f'{prefix}.{k}': v for k, v in params.items()}
    return params


def get_activations(model, inputs):
    # TODO implement
    pass


def get_gradients(model, inputs, outputs):
    # TODO implement
    pass


def prunable_modules(model, masked_modules, prune_classifier=True):

    modules = get_modules(model)
    prunable_modules = {name: module for name, module in modules.items()
                        if isinstance(module, masked_modules)}

    if not prune_classifier:
        clf = get_classifier_module(model)
        if clf in prunable_modules:
            del prunable_modules[clf]

    return prunable_modules


def fraction_to_keep(compression, model, prunable_modules):
    # return 1/compression # BYPASS FOR DEBUG
    # Compression = total / ( fraction * prunable + (total-prunable))
    # Using algrebra fraction is equal to
    # fraction = total/prunable * (1/compression - 1) + 1
    total_size = model_size(model)[0]
    # [1] is so we use nonzeros, this is useful for compounding
    prunable_size = sum([model_size(m)[1] for m in prunable_modules.values()])
    fraction = total_size / prunable_size * (1/compression - 1) + 1
    assert 0 < fraction <= 1, \
        f"Cannot compress to {1/compression} model with {(total_size-prunable_size)/total_size}" +\
        "fraction of unprunable parameters"
    return fraction
