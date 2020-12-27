# Emsemble model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyEnsemble(nn.Module):
    def __init__(self, midas, yolo_decoder, train_yolo=False, train_rcnn=False, train_all=True):
        super(MyEnsemble, self).__init__()
        self.midas = midas
        self.yolo_decoder = yolo_decoder
        self.train_yolo = train_yolo
        self.train_rcnn = train_rcnn
        self.train_all = train_all
        # Remove last linear layer
        # self.modelA.fc = nn.Identity()
        # self.modelB.fc = nn.Identity()
        
        # Create a 3 Layer buffer between Midas Encoder and Yolo Decoder
        # Everything would be freezed but this
        
        self.yolo_buffer = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )

        
    def forward(self, x):

        layer_1 = self.midas.pretrained.layer1(x.clone())
        layer_2 = self.midas.pretrained.layer2(layer_1)
        layer_3 = self.midas.pretrained.layer3(layer_2)
        layer_4 = self.midas.pretrained.layer4(layer_3)

        # if self.train_all:
        layer_1_rn = self.midas.scratch.layer1_rn(layer_1)
        layer_2_rn = self.midas.scratch.layer2_rn(layer_2)
        layer_3_rn = self.midas.scratch.layer3_rn(layer_3)
        layer_4_rn = self.midas.scratch.layer4_rn(layer_4)

        path_4 = self.midas.scratch.refinenet4(layer_4_rn)
        path_3 = self.midas.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.midas.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.midas.scratch.refinenet1(path_2, layer_1_rn)

        out = self.midas.scratch.output_conv(path_1)

        # Now adding a buffer and Yolo Layers
        yolo_buffer = self.yolo_buffer(layer_4)
        yolo_output = self.yolo_decoder(yolo_buffer)         
      

        return yolo_output, torch.squeeze(out, dim=1)
