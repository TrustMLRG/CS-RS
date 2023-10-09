# ******************************************************************************
import torch
from torchvision.models.resnet import resnet50,resnet18
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.ham_resnet import resnet as resnet_ham
from datasets import get_normalize_layer
from torch.nn.functional import interpolate

import pdb

ARCHITECTURES = ["resnet50","resnet18", "cifar_resnet20","cifar_resnet56", "cifar_resnet110"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)
    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if dataset == 'ham':
        model = resnet_ham().cuda()
        return model
    elif dataset == 'imagenette':
        model = resnet_ham().cuda()
        return model
    elif dataset == 'cifar10':
        if arch == "cifar_resnet20":
            model = resnet_cifar(depth=20, num_classes=10).cuda()
        elif arch == "cifar_resnet110":
            model = resnet_cifar(depth=110, num_classes=10).cuda()
        elif arch == "cifar_resnet56":
            model = resnet_cifar(depth=56, num_classes = 10).cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
    # return model


