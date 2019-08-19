
import torch

from .modules import masked_modules


def compute_masks(model, strategy):

    masks = {}

    for param, tensor in model.state_dict().items():
        if strategy.match(param):
            masks[param] = strategy.mask(tensor.detach().cpu().numpy())

    return masks


def apply_masks(model, masks):

    for param, tensor in model.state_dict().items():
        if param in masks:
            mask = torch.from_numpy(masks[param])
            if tensor.is_cuda:
                mask = mask.to(tensor.get_device())
            tensor.data.mul_(mask)
    return model


def masked_module(module, strategy):
    new_children = {}

    for name, submodule in module.named_children():
        if isinstance(submodule, tuple(masked_modules)):

            weight_mask = torch.from_numpy(strategy.mask(submodule.weight.detach().numpy()))
            if submodule.bias is not None:
                bias_mask = torch.from_numpy(strategy.mask(submodule.bias.detach().numpy()))
            else:
                bias_mask = None

            masked = masked_modules[type(submodule)](submodule, weight_mask, bias_mask)

            new_children[name] = masked
        else:
            # Recurse if children
            masked_module(submodule, strategy)
    # outside of loop otherwise the iterator will change
    for name, masked in new_children.items():
        setattr(module, name, masked)

    return module
