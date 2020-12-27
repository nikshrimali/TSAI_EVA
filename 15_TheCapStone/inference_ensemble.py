from inference_yolo import *
from inference_midas import *

def generate_output(model, midas=True, yolo=True):



    source = r'D:\Python Projects\EVA\15_TheCapStone\custom_data\images'

    if midas:
        run(input_path, output_path, model)
    
    if yolo:
        





