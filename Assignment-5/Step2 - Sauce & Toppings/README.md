# Making the pizza thin and cheezy!!!

My love for thin crusts and assignment marks made me do some experiments which frankly didn't dissappoint.

Below mentioned Sauses and toppings used

- Batch Normalization
- More Thinning of model
- Regularization

After thinning our base, we need to add some toppings to make our meal delicious, so lets start with adding some normalization in our code by adding a batchnormalization in each layer.


# Exp 1 - The BatchNorm

## Expectations

- Decreasing the params, might lead to a drop in accuracy

- BatchNorm normalizes our channels, increase their sharpness. Expecting better prediction

- In total not much expectations
### Results

- No of parameters - 14600
- Max training accuracy - 100%
- Max testing accuracy - 99.42%
- No of epochs used - 30


### Analysis

- Testing accuracy crosses 99.4% mark!!!!

- Model starts at a improved training accuracy - 96%

- Still far far away from goal (10k) - 14,800 (Additional parameters for BatchNorm)

- Training time has increased due to additional computations of batchnorm (from 30 sec to 6-8 sec per epochs to again 30-35 secs)

- There is a significant gap remains in training accuracy and test accuracy, but the difference is decreasing. 

- Since training accuracy has reached almost 100% there is no further scope of increasing the test accuracy 

- The model is convergs much faster at 12th epoch the training 

- Test accuracy still sees large values of ups and downs.

- Average loss shows has increased with the no of epochs

- By adding BatchNorm the model is showing performace equal to huge code (6 mill one)

- Post each epoch the model is gaining some training accuracy, but no change in the test accuracy, hence even decrasing the params will work

- This model is still overfitting. As training accuracy touches 100%, there can be no more improvements expected with this model in the testing accuracy.

***************************************************************************

# Exp 2 - More Thinning of model

Time to be realistic

> Trimming down the model to 7.8k, epochs to 14

- No of parameters - 7876
- No of epochs - 15
- Starting training accuracy - 94.91
- Starting training accuracy - 97.95
- Max training accuracy - 99.79%
- Max testing accuracy - 99.24%



### Analysis

- We reigned the wild stallion reduced params to a total of 7876

- There is a significant gap remains in training accuracy and test accuracy, but the difference is decreasing. The model seems to be overfitting still

- The model surprised by  getting 99.24 test accuracy in just 5th epoch, only to disappoint in later epochs. Model seems to be overshooting the minima

- Test accuracy still sees large values of ups and downs.

- Performance is like a pendulam, need to decrease the difference by adding some regularization


## Future enhancements

- Adding a regularizer (Dropout and Augmentation) to decrease difference between training and testing accuracy

- Decrease learning rate in later epochs to prevent model overshooting the minima

- Post each epoch the model is gaining some training accuracy, but no change in the test accuracy

***********************************************************

## Exp 3 - Regularization

Less difference between training and testing accuracy..

## Results

- No of parameters - 7876
- No of epochs - 15
- Dropout - *5%
- Starting training accuracy - 93.89
- Starting training accuracy - 98.28
- Max training accuracy - 99.33%
- Max testing accuracy - 99.29%

### Analysis

- Better accuracy than the last mode (Increased by .04%)

- Logs are beautiful as the training and testing accuracy are doing PDA with each other.( Awwww!!!!!)

- Overfitting problem seems to be resolved.

- Model can be pushed further since training accuracy is 99.33

- Swings in the graphs are still a concern, drops after rises (Looks like a Overshooting minima)



## Future Aspirations

- Learning rates experiment
- Augmentation to further regularize and train harder

**********************************************************************
