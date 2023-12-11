import torch.nn as nn
import os
import torch

# Common functions

def _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    bn = nn.BatchNorm2d(num_features=out_channels)
    return nn.Sequential(conv, bn)

def _conv2d_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
    conv2d_bn = _conv2d_bn(in_channels, out_channels, kernel_size, stride, padding)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv2d_bn, relu)

# ResNet Block definitions

class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downscale=False):
        super(_BasicBlock, self).__init__()
        self.down_sampler = None
        stride = 1
        if downscale:
            self.down_sampler = _conv2d_bn(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
            stride = 2
        self.conv_bn_relu1 = _conv2d_bn_relu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv_bn2 = _conv2d_bn(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        input = x
        if self.down_sampler:
            input = self.down_sampler(x)
        residual = self.conv_bn_relu1(x)
        residual = self.conv_bn2(residual)
        out = input + residual
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.batch_norm1(self.conv1(x))
        x = self.batch_norm2(self.conv2(x))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        return x

# ResNet Model definitions

class ResNetMini(nn.Module):
    def __init__(self, layers, num_classes=10):
        super(ResNetMini, self).__init__()
        self.conv1 = _conv2d_bn_relu(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.__make_layers(layers[0], in_channels=16, out_channels=16, downscale=False)
        self.layer2 = self.__make_layers(layers[1], in_channels=16, out_channels=32, downscale=True)
        self.layer3 = self.__make_layers(layers[2], in_channels=32, out_channels=64, downscale=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def __make_layers(self, num_layer_stack, in_channels, out_channels, downscale):
        layers = []
        layers.append(_BasicBlock(in_channels=in_channels, out_channels=out_channels, downscale=downscale))
        for i in range(num_layer_stack - 1):
            layers.append(_BasicBlock(in_channels=out_channels, out_channels=out_channels, downscale=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Loading pretrained weights

def _load_state_dict(model, arch, dataset):
    script_dir = os.path.dirname(__file__)
    dir = os.path.join(script_dir, "state_dicts", f"{arch}_{dataset}.pth")
    state_dict = torch.load(dir)
    model.load_state_dict(state_dict)
    return model

# Define the number of blocks for each ResNet variant

resnet_blocks = {
    "resnet20": [3, 3, 3],
    "resnet32": [5, 5, 5],
    "resnet56": [9, 9, 9],
    "resnet50": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
    "resnet152": [3, 8, 36, 3],
}

# Create ResNet models

def _make_resnet(num_classes, resnet_variant, pretrained, dataset):
    layers = resnet_blocks.get(resnet_variant, None)

    if layers is None:
        raise ValueError(f"Invalid ResNet variant: {resnet_variant}")

    if resnet_variant in ["resnet20", "resnet32", "resnet56"]:
        model = ResNetMini(layers=layers, num_classes=num_classes)
    else:
        model = ResNet(Bottleneck, layers, num_classes)
    
    if pretrained:
        model = _load_state_dict(model, resnet_variant, dataset)
    
    return model

def ResNet20(num_classes, pretrained=False, dataset="imagenette"):
    return _make_resnet(num_classes, "resnet20", pretrained, dataset)

def ResNet32(num_classes, pretrained=False, dataset="imagenette"):
    return _make_resnet(num_classes, "resnet32", pretrained, dataset)

def ResNet56(num_classes, pretrained=False, dataset="imagenette"):
    return _make_resnet(num_classes, "resnet56", pretrained, dataset)

def ResNet50(num_classes, pretrained=False, dataset="imagenette"):
    return _make_resnet(num_classes, "resnet50", pretrained, dataset)

def ResNet101(num_classes, pretrained=False, dataset="imagenette"):
    return _make_resnet(num_classes, "resnet101", pretrained, dataset)

def ResNet152(num_classes, pretrained=False, dataset="imagenette"):
    return _make_resnet(num_classes, "resnet152", pretrained, dataset)