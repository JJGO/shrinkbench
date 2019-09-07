from torch import nn
import torchvision.models


# TODO refactor replace_head and get_classifier_params into one func


def reduce_linear_layer(layer, n_classes, keep_weights=False):
    assert isinstance(layer, nn.Linear)
    new_layer = nn.Linear(layer.in_features, n_classes)
    if keep_weights:
        new_layer.weight.data = layer.weight.data[:n_classes, ...]
        new_layer.bias.data = layer.bias.data[:n_classes]
    return new_layer


def replace_head(model, n_classes, keep_weights=True):

    from .. import models as custom_models

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

    # classifier is .linear
    MODELS_WITH_LINEAR = (
        custom_models.cifar_resnet.ResNet
    )

    # classifier is .classifier[-1]
    MODELS_WITH_CLASSIFIER_LIST = (
        torchvision.models.AlexNet,
        torchvision.models.VGG,
        torchvision.models.MobileNetV2,
        torchvision.models.MNASNet,
    )

    # keep_weights is for imagenet subdatasets where
    # one samples the first N

    if isinstance(model, MODELS_WITH_FC):
        model.fc = reduce_linear_layer(model.fc, n_classes, keep_weights)

    elif isinstance(model, MODELS_WITH_LINEAR):
        model.linear = reduce_linear_layer(model.linear, n_classes, keep_weights)

    elif isinstance(model, MODELS_WITH_CLASSIFIER):
        model.classifier = reduce_linear_layer(model.classifier, n_classes, keep_weights)

    elif isinstance(model, MODELS_WITH_CLASSIFIER_LIST):
        model.classifier[-1] = reduce_linear_layer(model.classifier[-1],
                                                   n_classes, keep_weights)

    elif isinstance(model, torchvision.models.SqueezeNet):
        # Weird, uses convs
        raise NotImplementedError()

    elif isinstance(model, torchvision.models.Inception3):
        # Weird, expects 299 and aux outputs
        raise NotImplementedError()

    # TODO include efficientnet

    else:
        raise ValueError(f"Model {model} not recognized")


def get_classifier_module(model):
    from .. import models as custom_models

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

    # classifier is .linear
    MODELS_WITH_LINEAR = (
        custom_models.cifar_resnet.ResNet
    )

    # classifier is .classifier[-1]
    MODELS_WITH_CLASSIFIER_LIST = (
        torchvision.models.AlexNet,
        torchvision.models.VGG,
        torchvision.models.MobileNetV2,
        torchvision.models.MNASNet,
        custom_models.cifar_vgg.VGGBnDrop,
    )

    if isinstance(model, MODELS_WITH_FC):
        clf = 'fc'

    elif isinstance(model, MODELS_WITH_LINEAR):
        clf = 'linear'

    elif isinstance(model, MODELS_WITH_CLASSIFIER):
        clf = 'classifier'

    elif isinstance(model, MODELS_WITH_CLASSIFIER_LIST):
        i = len(model.classifier) - 1
        clf = f"classifier.{i}"

    elif isinstance(model, torchvision.models.SqueezeNet):
        # Weird, uses convs
        raise NotImplementedError()

    elif isinstance(model, torchvision.models.Inception3):
        # Weird, expects 299 and aux outputs
        raise NotImplementedError()

    # TODO include efficientnet

    elif isinstance(model, custom_models.mnistnet.MnistNet):
        clf = 'fc2'

    else:
        raise ValueError(f"Model {model} not recognized")

    return clf
