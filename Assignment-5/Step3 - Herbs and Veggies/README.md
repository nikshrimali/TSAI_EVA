# Herbs and Vegitables

*******************************************************************************************


As per our previous experiments, we created a model with parameters less than 8k, almost achieved the accuracy of 99.4%.

Now in the next notebooks, we will try to squeeze more out of our model to get consistent accuracy for same amount of parameters

Things to try

- 7*7 kernal
- Augmentation
- Variable learning rate
- Strides instead of maxpool
- 1*1 Convolutions
- Calculations of receptive field
- Global average pooling


### Analysis

### 7*7
Parameters have incresaed due to additional 7*7 parameters

Accuracy is somewhat low at about 99.35%

On trained for additional 15 epochs the results are

### Addition of GAP

- Accuracy starts at 85%
- Parameters now at 6k
- Final accuracy has taken a big hit, model is finding it difficult to breach the 99% test mark
- Loss is much larger than previous layers

- The models capacity is been reduced, adding 1 more layer to increase it and lets see the resutls. 


- No of parameters - 6894
- Max training accuracy - 98.67%
- Max testing accuracy - 98.7%
- No of epochs used - 15

 * Adding an additonal layer to see the results and reducing the dropouts to 5 % each layers
 
- No of parameters - 7914
- Max training accuracy - 98.67%
- Max testing accuracy - 98.7%
- No of epochs used - 15

* Addition of a 1*1 to transition layer has produced no significant results.

- No of parameters - 9326
- Max training accuracy - 98.12%
- Max testing accuracy - 98.55%
- No of epochs used - 15

Model starts at low accuracy, 60% training accuracy and 92.40% testing

* Reduced some parameters
- Maxpool is back in 1*1 trasition block


- No of parameters - 7586
- Starting Training accuracy - 60.99
- Starting Testing accuracy - 88.58
- Max training accuracy - 98.12%
- Max testing accuracy - 98.55%
- No of epochs used - 15

- Model shows resistance to cross 98.5% testing accuracymark even after training for 25 epochs
- 

* Adding an additonal layer after gap

- Expectations - Greater computational power at just 160 additional params
- More improvement in overall accuracy

Results

- No of parameters - 7586
- Starting Training accuracy - 64.16
- Starting Testing accuracy - 84
- Max training accuracy - 98.12%
- Max testing accuracy - 98.55%
- No of epochs used - 15
Results

- Accuracy reduced further instead of improving
- Final loss is much more (0.569)


Changing the learning rate to 0.05, removed 1*1 and added 3*3

Improvement in accuracy as 1*1 has been remoevd
Large learning rate, high crests and trumps

Results

- No of parameters - 7586
- Starting Training accuracy - 79.72
- Starting Testing accuracy - 95.07
- Max training accuracy - 98.72%
- Max testing accuracy - 98.83%
- No of epochs used - 15

No significant breakthrough achived.
Not able to cross earlier metrices

* Removed all the 2 strides
- Removed last layer of 1*1
- Replaced with 3*3
Increase in parameters
Results

- No of parameters - 7586
- Starting Training accuracy - 93.15
- Starting Testing accuracy - 98.37
- Max training accuracy - 99.4%
- Max testing accuracy - 99.41%
- No of epochs used - 15

Observations

Keeping large learning rate makes model learns faster, but needs to be decreased for the last epochs
Parameters are still large, need to remove 1 layer to at least be under 10k
Not much difference between training and testing accuracy

* Placed transition block with 1*1 as 3rd conv

- No of parameters - 7586
- Starting Training accuracy - 92.22
- Starting Testing accuracy - 98.37
- Max training accuracy - 99.4%
- Max testing accuracy - 99.41%
- No of epochs used - 15


- Added image augmentation

- No of parameters - 8662
- Starting Training accuracy - 90.78
- Starting Testing accuracy - 97.98
- Max training accuracy - 99.4%
- Max testing accuracy - 99.41%
- No of epochs used - 15



* Removing the augmentation on the test dataset

Expectations - Somepoints jump in the testing accuracy

- No of parameters - 8662
- Starting Training accuracy - 90.78
- Starting Testing accuracy - 97.98
- Max training accuracy - 99.4%
- Max testing accuracy - 99.41%
- No of epochs used - 15