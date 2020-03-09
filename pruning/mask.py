import numpy as np
import pandas as pd

from .modules import masked_modules, _ensure_tensor, _same_device, MaskedModule


def mask_module(module, masks):
    """Recursively mask a torch.nn Module

    Changes layers so that backprop doesn't get to masked parameters
    Note this operates inplace and modifies the passed network

    Arguments:
        module {torch.nn.Module} -- Module to mask
        masks Dict(torch.nn.Module : Dict(str:numpy.ndarray))
            -- Dictionary with masks for each weight tensor

    Returns:
        torch.nn.Module -- Same as id as input module, but after masking
    """

    # Need to store the new children so iteration won't break
    new_children = {}

    for name, submodule in module.named_children():

        if submodule in masks:
            mask_kwargs = {k+'_mask': v for k, v in masks[submodule].items()}
            if isinstance(submodule, MaskedModule):
                submodule.set_masks(**mask_kwargs)
            else:
                masked = masked_modules[type(submodule)](submodule, **mask_kwargs)
                new_children[name] = masked

        # Recurse for children
        mask_module(submodule, masks)

    # We replace the children outside of loop
    # otherwise the iterator will change
    for name, masked in new_children.items():
        setattr(module, name, masked)

    return module


def apply_masks(module, masks):
    """Recursively mask a torch.nn Module

    Zeros out masked parameters, does not change the layer
    Note this operates inplace and modifies the passed network

    Arguments:
        module {torch.nn.Module} -- Module to mask
        masks Dict(torch.nn.Module : Dict(str:numpy.ndarray))
            -- Dictionary with masks for each weight tensor

    Returns:
        torch.nn.Module -- Same as id as input module, but after masking
    """
    for name, submodule in module.named_children():

        if submodule in masks:

            for attr, mask in masks[submodule].items():
                param = getattr(submodule, attr)
                mask = _same_device(_ensure_tensor(mask), param)
                param.data.mul_(mask)

        # Recurse if children
        apply_masks(submodule, masks)

    return module


# Aux functions

def masks_details(model, masks):
    """Debug information for collection of masks

    Returns a dataframe with summary information of all masks

    Arguments:
        model {torch.nn.Module} -- torch module that the masks correspond to
        masks Dict(torch.nn.Module : Dict(str:numpy.ndarray))
            -- Dictionary with masks for each weight tensor

    Returns:
        pandas.DataFrame -- DataFrame with compression, size and shape for each mask
    """
    rows = []
    for name, module in model.named_modules():
        if module in masks:
            for k, v in masks[module].items():
                rows.append([name, k, 1/v.mean(), np.prod(v.shape), v.shape])
    columns = ['module', 'param', 'comp', 'size', 'shape']
    return pd.DataFrame(rows, columns=columns)
