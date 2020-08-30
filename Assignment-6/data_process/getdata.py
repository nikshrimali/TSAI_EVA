# Code for downloading the data
# import transformations
from torchvision import datasets
from torchvision import transforms

from data_process.transformations import *


# from transformations import *

transformations = GetTransforms()
train_transforms = transforms.Compose(transformations.trainparams())
test_transforms = transforms.Compose(transformations.testparams())

# train_transforms = [
#     transforms.RandomRotation((-14.0, 14.0), fill=(1,)),
#     transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,),(0.3081,))
#     ]

#         # return train_transformations

#     # def testparams(self):
# test_transforms = [
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,),(0.3081,))
# ]

class GetTrainData():
    def __init__(self, dir_name:str):
        self.dirname = dir_name

    def download_train_data(self):
        return datasets.MNIST('./data', train=True, download=True, transform=train_transforms)

    def download_test_data(self):
        return datasets.MNIST('./data', train=False, download=True, transform=test_transforms)