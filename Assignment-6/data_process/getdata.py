# Code for downloading the data
# import transformations
from torchvision import datasets

from data_process.transformations import *


# from transformations import *
print(help(GetTransforms))

transformations = GetTransforms()
train_transforms = transforms.Compose(transformations.train)
test_transforms = transforms.Compose(transformations.test)

class GetTrainData():
    def __init__(self, dir_name:str):
        self.dirname = dir_name

    def download_train_data(self):
        return datasets.MNIST('./data', train=True, download=True, transform=train_transforms)

    def download_test_data(self):
        return datasets.MNIST('./data', train=False, download=True, transform=test_transforms)