import numpy as np
import pandas as pd

from .modules import masked_modules, _ensure_tensor, _same_device
from ..models.head import get_classifier_module


def mask_module(module, masks):

    # Need to store the new children so iteration won't break
    new_children = {}

    for name, submodule in module.named_children():

        if submodule in masks:
            mask_kwargs = {k+'_mask': v for k, v in masks[submodule].items()}
            masked = masked_modules[type(submodule)](submodule, **mask_kwargs)
            new_children[name] = masked

        # Recurse for children
        mask_module(submodule, masks)

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


def masks_details(model, masks):
    rows = []
    for name, module in model.named_modules():
        if module in masks:
            for k, v in masks[module].items():
                rows.append([name, k, 1/v.mean(), np.prod(v.shape), v.shape])
    columns = ['module', 'param', 'comp', 'size', 'shape']
    return pd.DataFrame(rows, columns=columns)


def apply_masks(module, masks):
    for name, submodule in module.named_children():

        if submodule in masks:

            for attr, mask in masks[submodule].items():
                param = getattr(submodule, attr)
                mask = _same_device(_ensure_tensor(mask), param)
                param.data.mul_(mask)

        # Recurse if children
        apply_masks(submodule, masks)

    return module
