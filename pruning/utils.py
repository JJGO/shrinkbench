"""Auxiliary utils for implementing pruning strategies
"""

from collections import OrderedDict, defaultdict

import torch
from torch import nn

from ..models.head import get_classifier_module


def hook_applyfn(hook, model, forward=False, backward=False):
    assert forward ^ backward, \
        "Either forward or backward must be True"
    hooks = []

    def register_hook(module):
        if (
            not isinstance(module, nn.Sequential)
            and
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.ModuleDict)
            and
            not (module == model)
        ):
            if forward:
                hooks.append(module.register_forward_hook(hook))
            if backward:
                hooks.append(module.register_backward_hook(hook))

    return register_hook, hooks


# def get_modules(module, prefix=""):
#     """Recursively find all submodules from torch modules,
#     returning them in state_dict format
#     """
#     # TODO unnecesary given named_modules
#     modules = {}
#     for n, m in module.named_children():
#         modules[prefix+n] = m
#         modules.update(get_modules(m, prefix=prefix+n+'.'))
#     return modules


def get_params(model, recurse=False):
    params = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    return params


def get_activations(model, input):

    activations = OrderedDict()

    def store_activations(module, input, output):
        if isinstance(module, nn.ReLU):
            # TODO Fix. ResNet18 implementation reuses a
            # single ReLU layer?
            return
        assert module not in activations, \
            f"{module} already in activations"
        # TODO remove [0], not all models have a single input
        activations[module] = (input[0].detach().cpu().numpy().copy(),
                               output.detach().cpu().numpy().copy(),)

    fn, hooks = hook_applyfn(store_activations, model, forward=True)
    model.apply(fn)
    with torch.no_grad():
        model(input)

    for h in hooks:
        h.remove()

    return activations


def get_gradients(model, inputs, outputs):
    # TODO implement using model.register_backward_hook()
    # So it is harder than it seems, the grad_input contains also the gradients
    # with respect to the weights and so far order seems to be (bias, input, weight)
    # which is confusing
    # Moreover, a lot of the time the output activation we are looking for is the
    # one after the ReLU and F.ReLU (or any functional call) will not be called by
    # the forward or backward hook
    # Discussion here
    # https://discuss.pytorch.org/t/how-to-register-hook-function-for-functional-form/25775
    # Best way seems to be monkey patching F.ReLU & other functional ops
    # That'll also help figuring out how to compute a module graph
    pass


def get_param_gradients(model, inputs, outputs, loss_func=None, by_module=True):

    gradients = OrderedDict()

    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()

    training = model.training
    model.train()
    pred = model(inputs)
    loss = loss_func(pred, outputs)
    loss.backward()

    if by_module:
        gradients = defaultdict(OrderedDict)
        for module in model.modules():
            assert module not in gradients
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    gradients[module][name] = param.grad.detach().cpu().numpy().copy()

    else:
        gradients = OrderedDict()
        for name, param in model.named_parameters():
            assert name not in gradients
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy().copy()

    model.zero_grad()
    model.train(training)

    return gradients


def prunable_modules(model, masked_modules, prune_classifier=True):

    modules = model.named_modules()
    prunable_modules = {name: module for name, module in modules
                        if isinstance(module, masked_modules)}

    if not prune_classifier:
        clf = get_classifier_module(model)
        if clf in prunable_modules:
            del prunable_modules[clf]

    prunable_modules = list(prunable_modules.values())

    return prunable_modules


def fraction_to_keep(compression, model, prunable_modules):
    """ Return fraction of params to keep to achieve compression ratio

    Compression = total / ( fraction * prunable + (total-prunable))
    # Using algrebra fraction is equal to
    # fraction = total/prunable * (1/compression - 1) + 1
    """
    from ..metrics import model_size
    # return 1/compression # BYPASS FOR DEBUG
    # TODO fix for compounding
    total_size = model_size(model)[0]
    # [1] is so we use nonzeros, this is useful for compounding
    prunable_size = sum([model_size(m)[0] for m in prunable_modules])
    nonprunable_size = total_size - prunable_size
    fraction = 1 / prunable_size * (total_size/compression - nonprunable_size)
    assert 0 < fraction <= 1, \
        f"Cannot compress to {1/compression} model with {nonprunable_size/total_size}" + \
        "fraction of unprunable parameters"
    return fraction

# def fraction_to_keep(compression, model, prunable_modules):
#     """ Return fraction of params to keep to achieve compression ratio
#     total = prunable + nonprunable + zeros
#     Compression = (prunable + nonprunable) / (fraction * prunable + nonprunable)
#     # Using algrebra fraction is equal to
#     # fraction = (prunable + nonprunable) / (prunable * compression) - (nonprunable / prunable)
#     """

#     total_size, nonzero_size = model_size(model)
#     # [1] is so we use nonzeros, this is useful for compounding
#     prunable_size = sum([model_size(m)[1] for m in prunable_modules.values()])
#     nonprunable_size = nonzero_size - prunable_size

#     fraction = (prunable_size + nonprunable_size) / (prunable_size * compression) - (nonprunable_size / prunable_size)
#     assert 0 < fraction <= 1, \
#         f"Cannot compress to {1/compression} model with {(total_size-prunable_size)/total_size}" +\
#         "fraction of unprunable parameters"
#     print(f"{fraction:.2f} <> {prunable_size/total_size:.2f} {nonprunable_size/total_size:.2f}")
#     return fraction
