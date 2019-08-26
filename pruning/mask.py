from .modules import masked_modules


def get_modules(module, prefix=""):
    """Recursively find all submodules from torch modules,
    returning them in state_dict format
    """
    modules = {}
    for n, m in module.named_children():
        modules[prefix+n] = m
        modules.update(get_modules(m, prefix=prefix+n+'.'))
    return modules


def compute_masks(model, strategy):
    """Compute masks using the said strategy for every model
    This is a straight forward implementation that supports
    strategies that prune each module independently
    """
    masks = {}

    modules = get_modules(model)

    for name, module in modules.items():
        masks_ = strategy.module_masks(module)
        if len(masks_) > 0:
            masks[name] = masks_

    return masks


def mask_module(module, masks, prefix=""):

    # Need to store the new children so iteration won't break
    new_children = {}

    for name, submodule in module.named_children():

        if prefix + name in masks:
            mask_kwargs = {k+'_mask': v for k, v in masks[prefix + name].items()}
            masked = masked_modules[type(submodule)](submodule, **mask_kwargs)
            new_children[name] = masked

            # Recurse if children
            mask_module(submodule, masks, prefix=prefix+name+'.')

    # We replace the children outside of loop
    # otherwise the iterator will change
    for name, masked in new_children.items():
        setattr(module, name, masked)

    return module



    # for param, tensor in model.state_dict().items():
    #     if strategy.match(param):
    #         masks[param] = strategy.mask(tensor.detach().cpu().numpy())

    # return masks


# def apply_masks(model, masks):

#     for param, tensor in model.state_dict().items():
#         if param in masks:
#             mask = torch.from_numpy(masks[param])
#             if tensor.is_cuda:
#                 mask = mask.to(tensor.get_device())
#             tensor.data.mul_(mask)
#     return model


# def masked_module(module, strategy):
#     new_children = {}

#     for name, submodule in module.named_children():
#         if isinstance(submodule, tuple(masked_modules)):

#             weight_mask = torch.from_numpy(strategy.mask(submodule.weight.detach().numpy()))
#             if submodule.bias is not None:
#                 # TODO Fix for Channel Prunning
#                 bias_mask = torch.from_numpy(strategy.mask(submodule.bias.detach().numpy()))
#             else:
#                 bias_mask = None

#             masked = masked_modules[type(submodule)](submodule, weight_mask, bias_mask)

#             new_children[name] = masked
#         else:
#             # Recurse if children
#             masked_module(submodule, strategy)
#     # outside of loop otherwise the iterator will change
#     for name, masked in new_children.items():
#         setattr(module, name, masked)

#     return module




