import numpy as np
import pandas as pd

from .modules import masked_modules
from ..models.head import get_classifier_module


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


def unmask_classifier(model, masks):

    clf = get_classifier_module(model)

    if clf in masks:
        for p in masks[clf]:
            masks[clf][p] = np.ones_like(masks[clf][p])

    return masks


def masks_details(masks):
    rows = []
    for module, module_masks in masks.items():
        for k, v in module_masks.items():
            rows.append([module, k, 1/v.mean(), np.prod(v.shape), v.shape])
    columns = ['module', 'param', 'comp', 'size', 'shape']
    return pd.DataFrame(rows, columns=columns)

# from .utils import get_modules
# def compute_masks(model, strategy):
#     """Compute masks using the said strategy for every model
#     This is a straight forward implementation that supports
#     strategies that prune each module independently
#     """
#     masks = {}

#     modules = get_modules(model)

#     for name, module in modules.items():
#         masks_ = strategy.module_masks(module)
#         if len(masks_) > 0:
#             masks[name] = masks_

#     return masks
