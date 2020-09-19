# Assignment-8

 Submitted by -
> Nikhil Shrimali

## Target  
- Go through this repository: https://github.com/kuangliu/pytorch-cifar
- Extract the ResNet18 model from this repository and add it to your API/repo. 
- Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
- Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 

## Submission

I have trained model, summary and observations can be found below.
The code is completely modularized and shrinkai is used where the source code exists

> Link to New Awesomeness - <a href= "https://github.com/nikshrimali/shrinkai"> Visit SHRINKAI</a>

#### Results

- No of parameters - 11,173,962
- No of epochs - 35
- Dropout - 10%
- Starting training accuracy - 42.37
- Starting testing accuracy - 55.24
- Max training accuracy - 93.94%
- Max testing accuracy - 91.25%

#### Obervations

- Model was largely overfitting (Max Train Acc: 100% Max Test Acc: 85%), hence I added dropout of 10% and image augmentations. Now model is slightly overfitting.

- Achitecture used is Resnet18, I have added log_softmax as without that, the loss values were going negative and I am using NLL Loss

- I had noticed that keeping learning rate as 0.01 trains the model good for 25 epochs, but model shows convergance, hence used StepLR to reduce learning rate by 10% after 25 epochs

## Future Aspirations

- I have added the image augmentations, but need more intution on what augmentations works on which dataset

# About RESNET18

- Resnet18 comes from the family of amazing architectures which is known as RESIDUAL Networks. The core idea of ResNet is introducing a so-called “identity shortcut connection” that skips one or more layers.A residual block is displayed as the following:
![Resnet18 Arch](./assets/resnet_arch.PNG)

- It has skip connections that allows it to have multiple receptive fields which makes it an amazing model and allows it idenfity object of every size
![Residual Block](./assets/residual-block.PNG)

- It has 1 convolution layer of 7x7 sized kernel (64), with a stride of 2
- It is followed by MaxPooling. In fact, ResNet has only 1 MaxPooling operation!
- It is followed by 4 ResNet blocks (config: 2,2,2,2)
- The channels are constant in each block (64, 128, 256, 512 respectively). Each block has only 3x3 kernels.
- The channel size is constant in each block
- Except for the first block, each block starts with a 3x3 kernel of stride 2 (this handles MaxPooling)
