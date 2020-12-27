#Load the weights from pre-trained model
import os
os.chdir("./models_all/YoloV3/")
print(os.getcwd())
from models import *
from utils.parse_config import *

def get_yolo():

    path = './cfg/yolov3-custom.cfg'
    yolo_model = Darknet(path).to("cuda")

    sc_yolokeys = list(yolo_model.state_dict().keys())
    model_path = r".\pretrained_model\best273.pt"
    ptmodel = torch.load(model_path)['model']
    pt_yolokeys = (list(ptmodel.keys())[330:])
    # Updating the Yolo model with Pre-trained weights
    for i, keys in enumerate(sc_yolokeys):
        yolo_model.state_dict()[keys] = ptmodel[pt_yolokeys[i]]
    # Freeze the layers for training
    for param in yolo_model.parameters():
        param.requires_grad = False
    return yolo_model