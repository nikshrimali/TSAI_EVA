
# The Kedgeree

> Submitted by - Nikhil Shrimali


I have modified the architecture of all 3 models to extract the decoder layers.

- Midas Model - Used as it is (Encoder + Decoder). Download whole MidasNet from Github repo from <a href = 'https://github.com/intel-isl/MiDaS'> Here</a>
- Yolo Model - Decoder Layers only. Git repo for only the decoder layers can be found <a href = 'https://github.com/nikshrimali/YoloV3'> Here</a>
- Planercnn - Decoder Layers only Git repo extracting the whole planercnn layers can be found <a href = 'https://github.com/NVlabs/planercnn'> Here</a> (Work in Progress)

A Colab notebook combining all solution can be found here <a href = 'https://colab.research.google.com/drive/16DHBysV3VH2yxVVkfPYAnovkX0hFKgDO?usp=sharing'> Here</a>


## Training the Model (Yolo Layer only)

> Currently only Yolo branch is available for training, Midas branch is being trained. Planercnn is still work in progress.
> Please note that by default encoders and decoders params are frozen, you can unfreeze them by looping and setting require_grad = True before sending them to Ensemble model class

- Colab notebook showing training can be found at <a href = 'https://github.com/miki998/YoloV3_Annotation_Tool'>YoloV3 Annotation Tool</a>
  
- For training Yolo Branch, we need to get bounding boxes data
- Annotate the data using the  <a href = 'https://github.com/miki998/YoloV3_Annotation_Tool'>YoloV3 Annotation Tool</a>
- Paste your images and labels into directory custom_data/images and labels into custom_data/labels

- I have utilized the <a href = 'https://drive.google.com/file/d/1wyjIOERwuAatnaXkNWVH59ojf_-7M7D8/view?usp=sharing'>YoloV3 Pretrained PPE Model</a> trained 273 epochs which can be downloaded frpm here. Paste it to YoloV3/Pretrained_model directory

## Inferencing on Model - Yolo + Midas

 
  - train_yolo - If set to true trains only the Yolo buffer layers and outputs only yolo layers
  - train_rcnn - If set to true trains only the Rcnn buffer layers and outputs only rcnn layers
  - train_all - Train buffer layers of both the networks and outputs consists of output of all 3 models


## Work Finished

- I am able to create ensemble model with Midas and Yolo Layers
- Each layer can be trained with setting the parameters in model
- Able to connect both and generate outputs
- I am also able to train the model (Yolo Buffer Layers)
- All previous issues related to testing pipeline of the Yolo output model are resolved
- 

## Work Pending

- Making code as generic as possible, currently most variables and paths are hardcoded
- Combine RCNN decoder into the model
- Training RCNN model on custom dataset