# Thinning the Base

This code is a diet version of base code, means we will trim down the no of 
parameters to study the behaviour of model on constrained resources.

Only modification to the model architecture is made in this.

## Results  

- No of parameters - 14600
- Max training accuracy - 99.61%
- Max testing accuracy - 99.06%
- No of epochs used - 30


## Analysis
- Model starts at a very low training accuracy - 72%

- The model is using too many parameters  Total - 14,600

- Huge reduction in training time (from 30 sec to 6-8 sec per epochs)

- There is a significant gap remains in training accuracy and test accuracy (Almost .6 - .7 %)

- Since training accuracy has reached almost 100% there is no further scope of increasing the test accuracy 

- Test accuracy still sees large values of ups and downs.

- Average loss shows has increased with the no of epochs

- The model is showing reluctance to cross the 99% mark even after increasing the epochs by 10, this shows the model is not capable of going any further

- Post each epoch the model is gaining some training accuracy, but no change in the test accuracy

