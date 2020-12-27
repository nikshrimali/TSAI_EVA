import torch
import torch.nn as nn

# Make encoder by loading pre-trained resnext101 model
def _make_encoder(features, use_pretrained):
    '''
    args:
        features () - Output value 256 default for pre-trained model
        use_pretrained (    ) - 
    '''
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    scratch = _make_scratch([256, 512, 1024, 2048], features)

    return pretrained, scratch

# Making the resnet backbone - Loading resnet weights???
def _make_resnet_backbone(resnet):
    '''
    Loads a pretrained models - Run this commands and see for yourself.
    '''
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained

# Download the resnext 101 model and returens it
def _make_pretrained_resnext101_wsl(use_pretrained):
    '''
    Args:
    use_pretrained - bool value
    '''
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)

# Scratch - 4 conv layers of shape as provided in list values
def _make_scratch(in_shape, out_shape):
    '''
    Args:
    in_shape (list): list of 4 that specifies input shape -  [256, 512, 1024, 2048]
    out_shape (int): Int value that specifies output shape
    '''
    scratch = nn.Module()
    # Byt why a constant output shape but different input shape
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    return scratch

# basically just a class around nn.functional.interpolate
class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x

# Normal resnet block of two convolution layers and relu operation
# Why this is requried when pre-trained model is already loaded - Is this part of decoder??
class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

# Expands the network into the  resnet blocks
# Creates two conv layer each time ResidualConvUnit is called, the output is summed up at the end
class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features - 256 output features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
    # What is xs? A list?
    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output
