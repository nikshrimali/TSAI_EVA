# Loading the pretrained encoder of Midas
import os
print(os.getcwd())
os.chdir(os.path.dirname(os.getcwd()))
print(os.getcwd())
os.chdir(r'.\midas')


from midas.midas_net import *
def get_midas():
    model_path = r"D:\Python Projects\EVA\15_TheCapStone\models_all\midas\model-f6b98070.pt"
    midas_model = MidasNet(model_path, non_negative=True).to("cuda")
    # Freezing layers for updates
    for param in midas_model.parameters():
        param.requires_grad_(False)
    return midas_model