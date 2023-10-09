from torchvision import transforms, datasets
from torchvision.datasets import MNIST
from typing import *
import torch
import os
from torch.utils.data import Dataset
import pdb
from process_HAM10k import load_ham_data 
# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

# list of all datasets
DATASETS = ["imagenette", "cifar10","mnist","ham"]

def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenette":
        return _imagenette(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "mnist":
        return _mnist(split)
    elif dataset == 'ham':
        return _ham(split)

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenette":
        return 10 # original:1000
    elif dataset == "cifar10":
        return 10
    elif dataset == 'imagenet':
        return 10
    elif dataset =='ham':
        return 4

def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenette":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

 

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

def _mnist(split:str) ->Dataset:
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if split == 'train':
        trainset = MNIST(root = './dataset_cache', train=True, download=True, transform=transform_train)
        return trainset
    elif split == 'test':
        testset = MNIST(root = './dataset_cache', train=False, download=True, transform=transform_test)
        return testset
   
def _imagenet(split: str) -> Dataset:

    dir = '/home/c01yuxi/CISPA-projects/certification_robustness-2022/macer_img/data/imagenet/imagenet-10'
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)

def _imagenette(split: str) -> Dataset:

    extra_size = 32
    image_size = 160
    dir = '/home/c01yuxi/CISPA-projects/certification_robustness-2022/smoothing-master/imagenette2-160'
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
             transforms.RandomResizedCrop(image_size, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform =  transforms.Compose([
            transforms.Resize(image_size + extra_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        # pdb.set_trace()
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds

def _ham(split):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225] )
    transform = transforms.Compose(
    [
    transforms.Resize(299), #299
    transforms.CenterCrop(299), #299
    transforms.ToTensor(),
    normalize
     ]
     )
    trainset, testset = load_ham_data(transform)
    if split == 'train':
        return trainset
    elif split == 'test':
        return testset


