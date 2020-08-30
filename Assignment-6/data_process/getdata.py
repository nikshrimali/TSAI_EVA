# Code for downloading the data
from torchvision import datasets
from torchvision import transforms

from data_process.transformations import *


# from transformations import *

transformations = GetTransforms()
train_transforms = transforms.Compose(transformations.trainparams())
test_transforms = transforms.Compose(transformations.testparams())

class GetTrainData():
    def __init__(self, dir_name:str):
        self.dirname = dir_name

    def download_train_data(self):
        return datasets.MNIST('./data', train=True, download=True, transform=train_transforms)

    def download_test_data(self):
        return datasets.MNIST('./data', train=False, download=True, transform=test_transforms)