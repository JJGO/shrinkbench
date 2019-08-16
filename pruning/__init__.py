import torch


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
