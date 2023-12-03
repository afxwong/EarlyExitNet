'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, self.expansion * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion * growth_rate)
        self.conv2 = nn.Conv2d(self.expansion * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, block_type, nblocks, growth_rate=12, reduction=0.5, num_classes=10, ee_layer_locations=[]):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.num_classes = num_classes

        num_planes = 2 * growth_rate
        if num_classes == 1000:
            self.conv1 = nn.Sequential(nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False),
                                       nn.BatchNorm2d(num_planes),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block_type, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block_type, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes
        
        self.dense3 = self._make_dense_layers(block_type, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes
        
        self.dense4 = self._make_dense_layers(block_type, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block_type, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block_type(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        
        out = self.dense4(out)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out
    
def densenet121(num_classes, pretrained=False, dataset="cifar100"):
    model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=num_classes)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            os.path.join(script_dir, "state_dicts", f"densenet121_{dataset}.pth"), map_location="cpu"
        )
        model.load_state_dict(state_dict)
    return model