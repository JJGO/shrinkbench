"""Auxiliary module for dealing with classifier layers in networks
"""
from torch import nn
import torchvision.models

# classifier is .fc
MODELS_WITH_FC = (
    torchvision.models.ResNet,
    torchvision.models.GoogLeNet,
    torchvision.models.ShuffleNetV2,
)

# classifier is .classifier
MODELS_WITH_CLASSIFIER = (
    torchvision.models.DenseNet,
)


# classifier is .classifier[-1]
MODELS_WITH_CLASSIFIER_LIST = (
    torchvision.models.AlexNet,
    torchvision.models.VGG,
    torchvision.models.MobileNetV2,
    torchvision.models.MNASNet,
)


def reduce_linear_layer(layer, n_classes, keep_weights=False):
    """Gets a torch.nn.Linear layer and resizes it to a new number of classes

    The keep_weights argument is useful for experiments that subset an
    existing dataset. E.g. First 10 or 100 classes of ImageNet.
    This is helpful when debugging pipelines

    Arguments:
        layer {nn.Linear} -- Dense Classifier Layer
        n_classes {int} -- New Number of layers

    Keyword Arguments:
        keep_weights {bool} -- Whether to keep previous weights (default: {False})

    Returns:
        [nn.Linear] -- New Dense layer for classification
    """
    assert isinstance(layer, nn.Linear)
    new_layer = nn.Linear(layer.in_features, n_classes)
    if keep_weights:
        new_layer.weight.data = layer.weight.data[:n_classes, ...]
        new_layer.bias.data = layer.bias.data[:n_classes]
    return new_layer


def replace_head(model, n_classes, keep_weights=True):
    """Replace the classifier layer with a different one

    This is needed when repurposing an architecture for a different task
    For example using an ImageNet designed ResNet 50 for Places365

    Arguments:
        model {nn.Module} -- Classification Network
        n_classes {int} -- New number of classes

    Keyword Arguments:
        keep_weights {bool} -- Whether to keep previous weights. See reduce_linear_layer (default: {True})

    Raises:
        NotImplementedError -- Raised when the architecture class is not supported
    """

    # keep_weights is for imagenet subdatasets where
    # one samples the first N

    if isinstance(model, MODELS_WITH_FC):
        model.fc = reduce_linear_layer(model.fc, n_classes, keep_weights)

    elif isinstance(model, MODELS_WITH_CLASSIFIER):
        model.classifier = reduce_linear_layer(model.classifier, n_classes, keep_weights)

    elif isinstance(model, MODELS_WITH_CLASSIFIER_LIST):
        model.classifier[-1] = reduce_linear_layer(model.classifier[-1],
                                                   n_classes, keep_weights)

    elif isinstance(model, torchvision.models.SqueezeNet):
        # TODO: Non standard, uses convs
        raise NotImplementedError()

    elif isinstance(model, torchvision.models.Inception3):
        # TODO: Non standard, expects 299 and aux outputs
        raise NotImplementedError()

    # TODO include efficientnet

    else:
        raise NotImplementedError(f"Model {model} not supported")


def get_classifier_module(model):
    """Get classifier module for many Vision Classification networks


    Arguments:
        model {nn.Module} -- Network to extract the classifier from

    Returns:
        [nn.Linear] -- Dense Layer that performs the last step of classification

    Raises:
        NotImplementedError -- Raised when the architecture class is not supported
    """

    if isinstance(model, MODELS_WITH_FC):
        clf = 'fc'

    elif isinstance(model, MODELS_WITH_CLASSIFIER):
        clf = 'classifier'

    elif isinstance(model, MODELS_WITH_CLASSIFIER_LIST):
        i = len(model.classifier) - 1
        clf = f"classifier.{i}"

    elif isinstance(model, torchvision.models.SqueezeNet):
        # TODO: Non standard, uses convs
        raise NotImplementedError()

    elif isinstance(model, torchvision.models.Inception3):
        # TODO: Non standard, expects 299 and aux outputs
        raise NotImplementedError()

    # TODO include efficientnet

    else:
        raise NotImplementedError(f"Model {model} not recognized")
    clf = getattr(model, clf)
    return clf


def mark_classifier(model):
    # Mark it manually for torchvision models
    clf_module = get_classifier_module(model)
    clf_module.is_classifier = True
    return model
