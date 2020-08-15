# Assignment 4 Submission

> Submitted by - Nikhil Shrimali, Balaji B, Manivel Sethu

## Target:
* 99.4% validation accuracy
* Less than 20k Parameters
* You can use anything from above you want. 
* Less than 20 Epochs
* No fully connected layer
* To learn how to add different into our model (BatchNorm, Dropouts etc)

## What worked for us

* To keep no of parameters in check, no of channels were kept below 20
* Every convolution layer were added with Batch Normalization and Dropout (the lazy way!!)
  - Batchnorm normalizes the images of complete batch to ur channels hence features can be  clearly found in the images
  -  Dropout is a very lazy and brutal way of removing some % age of nodes, hence has a regularization effect on the model, when some primary nodes are dropped, it forces other nodes to train to improve the accuracy
* Didn't commit same mistake as the legendary VGG16 by adding fully connected layers
* We used our Ant-Man which helped us with following:
  - Lesser computation requirement for reducing the number of channels 
  -  Use of existing channels to create complex channels (instead of re-convolution) 

* For effective and easier decision making we used Relu activation function into our model
* We didn't use any Flatten layer hence no spatial information was lost in our code

## What didn't worked for us
 * We also tried using Global average pooling but were not able to achieve the required accuracy
 * Going the traditional way by starting with 32 channels, we were overshooting the parameters no by 500%

 ## Future aspirations
 * Adding image augmentations
 * Replacing the Dropouts with cutouts
 * Replacing the Maxpool2D with Convolution layers with Strides = 2

