import torch
from torch import nn
from torch.functional import F
DROPOUT = 0.05

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,padding=1),  # 30
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(DROPOUT),

            nn.Conv2d(32, 64, 3, padding=1),  # 24
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.03),
            nn.MaxPool2d(2,2), # 12
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),  # 24
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.03),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, padding=1),  # 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
            nn.MaxPool2d(2,2), # 12

        )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, padding=0, dilation=2),  # 8
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.Dropout(DROPOUT),
        #     # nn.MaxPool2d(2,2),  # 7
        # )

        self.dpthwise_sep3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, groups=128, padding=1),  # 6
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),

            nn.Conv2d(128, 128, 1, groups=128),  # 4
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),

            nn.Conv2d(128, 128, 3, padding=1),  # 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),
            # nn.MaxPool2d(2,2),  # 7

            nn.Conv2d(128, 128, 3, padding=1),  # 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(DROPOUT),

            nn.AvgPool2d(kernel_size=4),

            nn.Conv2d(128, 10, 1),  #1

        )


    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.dpthwise_sep3(x)
        x = self.block4(x)
        x = x.view(-1,10)
        return F.log_softmax(x, dim=-1)
