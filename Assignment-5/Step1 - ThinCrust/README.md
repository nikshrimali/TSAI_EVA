# Thinning the Base

At TSAI, we have trainined enough neural networks, let me explain my work by my recently acquired skill of pizza making.

In our previous model, the pizza has a thick crust (6 Mill params!!! :o), but its 2020, I like my crust to be thin, light and juicy so lets trim the ingredients(params). Lets call it DietCode.

> Only modification to the model architecture is made in this. I will  be testing this on 30 epochs to study the behaviour better

## Expectations

- Drop in overall accuracy as we will trim the no of parameters by a very large margin

- Faster training times

- Decrease in the difference between  training and testing accuracy, too many parameters for such small dataset might lead to overfitting


## Results  

- No of parameters - 14600
- Starting training accuracy - 72.36
- Starting testing accuracy - 95.99
- Max training accuracy - 99.83%
- Max testing accuracy - 99.07%
- No of epochs used - 30


## Moment of Truth

- Model starts at a very low training accuracy - 72%

- The model is using too many parameters  Total - 14,600 Almost (4600 from our realistic dreams!!!)

- Huge reduction in training time (from 30 sec to 13-15 sec per epochs)

- There is a significant gap remains in training accuracy and test accuracy (Almost .6 - .7 %)

- Since training accuracy has reached almost 100% there is no further scope of increasing the test accuracy 

- Test accuracy still sees large values of ups and downs.

- Average loss shows has increased with the no of epochs

- The model is showing reluctance to cross the 99.83% mark even after increasing the epochs by 15, this shows even pushing model harder wont get us to 99.4 testing.

- Post each epoch the model is gaining some training accuracy, but no change in the test accuracy. These are clear signs of Overfitting!!!

- The testing accuarcy graph looks more like my career graph. We need to smoothing it out. Both of them!!!!

## Future Plans

- Decreasing Parameters further more
- Adding BatchNorm to sharpen the features in channels to get better accuracy
- Adding a regularization to decrease the difference between training and the testing accuracy

