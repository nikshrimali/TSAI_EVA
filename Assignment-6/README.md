# Assignment-6

 Submitted by - 

## Target  

GBN Note: read the paper. Please make sure you are not using the same batch size for BN and GBN jobs below. 

- Your assignment 6 is to take your best 5th code, and run bellow versions for 25 epochs and report findings:
  - with L1 + BN
  - with L2 + BN
  - with L1 and L2 with BN
  - with GBN
  - with L1 and L2 with GBN

- You cannot be running your code 5 times manually (-500 points for that). You need to be smarter and write a single loop or iterator to iterate through these conditions.
- draw ONE graph to show the validation accuracy curves for all 5 jobs above. This graph must have proper legends and it should be clear what we are looking at.
- draw ONE graph to show the loss change curves for all 5 jobs above. This graph must have proper legends and it should be clear what we are looking at.
- find any 25 misclassified images (combined into single image) for "with GBN" model. You should be using the saved model from the above jobs.  You MUST show the actual and predicted class names.
the explanatory README file that explains what is your code all about, your findings, and your single image showing 25 misclassified images.
- submit the Github link
 
Here are some of the questions for S6-Assignment-Solution:

- Upload the Validation Accuracy Change Graph (all 5 models combined) - 100 pts
- Upload the Loss Change Graph (all 5 models combined) - 100 pts
- Upload the image showing 25 misclassified images for the "with GBN" model. - 250 pts
- Explain your observation w.r.t. L1 and L2's performance in the regularization of your model. - 50 pts

## Our Submission

We have trained 5 models, details are as below


  - gbn - Model with Ghost Batch Norm(GBN)
  - l1_l2_gbn - Model with L1, L2 and GBN
  - 'l1_bn - Model with only L1 Regularization
  - l2_bn - Model with only L2 Regularization
  - l1_l2_bn - Model with both L1 and L2 Regularization


## The Ghost BatchNorm

It is proven to improve the performance of the existing models by some points, as the batch gets further divided into smaller batches and hence known to provide a regularizing effect the the model

We have tried the model with no of splits = (4, 8, 16) and batch size used for this is 256 instead of 128.

### No of Splits - 16

#### Results

- No of parameters - 7076
- No of epochs - 15
- Dropout - *3%
- Starting training accuracy - 78.40
- Starting testing accuracy - 97.28
- Max training accuracy - 98.36%
- Max testing accuracy - 99.22%

#### Obervations

The model is not able to achive the accuracy of our last model which was < 99.5%

GBN is known to have a regularization effect on the 

### No of Splits - 8 and Variable Learning Rate

> GBN can also have a regularizing effect

#### Results

- No of parameters - 7076
- No of epochs - 15
- Dropout - *3%
- Starting training accuracy - 80.56
- Starting testing accuracy - 97.93
- Max training accuracy - 98.53%
- Max testing accuracy - 99.39%

#### Obervations

- Increase in training and the testing (~.5%) accuracy 
The model is not able to achive the accuracy of our last model which was < 99.5%
- Training accuracy is continuously increasing, but the effect is not visible on the testing accuracy, this could be the case that we are regularizing too  much, since GBN can also have a regularizing effect.

- Model might be underfitting

- Also many ups and downs in the testing curves, the learning rate seems to be too high

### No of Splits - 4 and Less Image Augmentation

> When we add a new regularizer, to balance it off, we should also decrease the old ones, otherwise this could change the loss curve. Only small changes are required at a time.

#### Results

- No of parameters - 7076
- No of epochs - 15
- Dropout - *3%
- Starting training accuracy - 83.12
- Starting testing accuracy - 97.79
- Max training accuracy - 98.93%
- Max testing accuracy - 99.43%

#### Obervations

- Increase in starting training and decrease testing in (~.5%) accuracy 

- The max training and testing accuracy has not much changes, but testing accuracy curve is consistent in the final epochs, hence this looks like an improvement over the previous models


