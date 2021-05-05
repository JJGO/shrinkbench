import torch

def pretrained_weights(model):
    url = f'https://raw.githubusercontent.com/JJGO/shrinkbench-models/master/cifar10/{model}.th'
    print(url)
    return torch.hub.load_state_dict_from_url(url, map_location='cpu')
