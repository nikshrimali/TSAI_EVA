# Adding Toppings

We would be studying two versions hence two parts of Results and Analysis for a single notebook

After thinning our base, we need to add some toppings to make our meal delicious, so lets start with adding some normalization in our code by adding a batchnormalization in each layer.

## Adding BatchNormalization

BatchNorm normalizes our channels, like so that they can do a better prediction, hence expecting a jump in test set accuracy

Adding dropouts provides a regulizer effect to the channels, hence we can see the difference between training accuracy and testing accuracy to decrease

### Results  

- No of parameters - 14600
- Max training accuracy - 100%
- Max testing accuracy - 99.42%
- No of epochs used - 30


### Analysis

- Model starts at a very low training accuracy - 96% training accuracy

- The model is using too many parameters  Total - 14,800 (Additional parameters for BatchNorm)

- Training time has increased due to additional computations of batchnorm (from 30 sec to 6-8 sec per epochs to again 30-35 secs)

- There is a significant gap remains in training accuracy and test accuracy, but the difference is decreasing. Testing accuracy crosses 99.4% mark

- Since training accuracy has reached almost 100% there is no further scope of increasing the test accuracy 

- The model is converging much faster at 12th epoch the training 

- Test accuracy still sees large values of ups and downs.

- Average loss shows has increased with the no of epochs

- By adding BatchNorm the model is showing performace equal to huge code

- Post each epoch the model is gaining some training accuracy, but no change in the test accuracy

## Adding Dropout

The gap between the training and the test accuracy still seems huge, hence lets add some regularization by introducing the dropouts

Adding a modest dropout of 25% after each layer to study the effects

- No of parameters - 7896
- Max training accuracy - 98.02%
- Max testing accuracy - 98.86%
- No of epochs used - 15

- Loss is not decreasing much in the final layers

- Model seems to be converged at 6-7 epoch

## Exp-3

Getting highest accuracy when dropout is 10% added to every layer, since adding 25% to all layers was kinda hard on the model.
Hence reduced the Dropout a bit to experiment

- No of parameters - 7896
- Max training accuracy - 99.12%
- Max testing accuracy - 99.37%
- No of epochs used - 15

Model shows promise, can be pushed further

## Future Aspirations

- Global average pooling at the final layers

- Strides of 2 instead of maxpool

- 