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