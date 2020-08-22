

## Exp-1 - LR

Now after preparing the base and stuffing, we have to bake the pizza in extreme heat. I am putting my model inside a preheated (180 deg), so that it learns to work in hard conditions.

Adding the LR to tame the wild changes in the final epochs

> LR would be reduced by 10% after 10 epochs

## Results

- No of parameters - 7876
- No of epochs - 15
- Dropout - *5%
- Starting training accuracy - 93.89
- Starting testing accuracy - 98.46
- Max training accuracy - 99.47%
- Max testing accuracy - 99.35%

### Analysis


- Better training and testing accuracy than the last mode (Increased by .06%)

- Logs are beautiful as the training and testing accuracy are doing PDA with each other.( Awwww!!!!!)

- Model can be pushed further since training accuracy is 99.45

- Swings concern is resolved!!!


## Future Aspirations

- Augmentation to push

**********************************************************************
## Exp-2 - Adding Augmentations

Expecting that last push in the training accuracy
Might lead to some difference in training and  testing acc
Gap in training and testing accuracy,  testing acc would more than training

## Results

- No of parameters - 7876
- No of epochs - 15
- Dropout - *5%
- Starting training accuracy - 92.56
- Starting testing accuracy - 98.16
- Max training accuracy - 99.47%
- Max testing accuracy - 99.35%

### Analysis

- Fell short of the accuracy mark by a very less margin
- Testing  accuracy more than training accuracy
- Testing graph is linear, there are no crests and drops



## Future Aspirations

- Relook at the architecture, Receptive field

**********************************************************************
