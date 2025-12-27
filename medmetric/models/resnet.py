import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from dataclasses import dataclass
from typing import Optional, Sequence, Type, Union

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200',
    # MedicalNet backbone + pooling wrappers
    'MedicalNetResNet', 'MedicalNetResNetPooled', 'medicalnet_resnet'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MedicalNetResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(MedicalNetResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Return feature map (B, C, D', H', W') after layer4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Backwards-compatible function names; not required by extractor.
# These now build the backbone feature model instead of the seg head.
def ResNet(block, layers, **kwargs):
    shortcut_type = kwargs.get("shortcut_type", "B")
    no_cuda = kwargs.get("no_cuda", False)
    return MedicalNetResNet(block, layers, shortcut_type=shortcut_type, no_cuda=no_cuda)


def resnet10(**kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


class MedicalNetResNetPooled(nn.Module):
    """MedicalNet backbone + global average pooling + flatten.

    Forward returns:
      - features: (B, F)
      - optionally feature_map: (B, C, D', H', W') if return_map=True
    """

    def __init__(self, block, layers, return_map=False, shortcut_type='B', no_cuda=False):
        super().__init__()
        self.backbone = MedicalNetResNet(block, layers, shortcut_type=shortcut_type, no_cuda=no_cuda)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.return_map = return_map

    def forward(self, x):
        fmap = self.backbone(x)
        feats = self.avgpool(fmap).flatten(1)
        if self.return_map:
            return feats, fmap
        return feats


def medicalnet_resnet(
    block,
    layers,
    pooled=True,
    return_map=False,
    shortcut_type='B',
    no_cuda=False,
):
    """Factory for MedicalNet ResNet.

    Args:
        block: BasicBlock/Bottleneck string ('BasicBlock'/'Bottleneck').
        layers: list/tuple of 4 ints (e.g. [3,4,6,3]).
        pooled: if True returns pooled features (B,F), else returns feature map (B,C,D',H',W').
        return_map: only used if pooled=True; return (feats, fmap).
        shortcut_type: 'A' or 'B' (MedicalNet convention).
        no_cuda: used for shortcut type 'A' downsample in original code.
    """

    if isinstance(block, str):
        b = block.strip().lower()
        if b in ("basicblock", "basic"):
            block = BasicBlock
        elif b in ("bottleneck",):
            block = Bottleneck
        else:
            raise ValueError(
                f"Unknown block='{block}'. Expected 'BasicBlock' or 'Bottleneck' (or the class itself)."
            )

    if not isinstance(layers, (list, tuple)) or len(layers) != 4:
        raise ValueError(f"layers must be a list/tuple of length 4, got {layers!r}")
    layers = [int(x) for x in layers]

    if pooled:
        return MedicalNetResNetPooled(
            block=block,
            layers=layers,
            return_map=return_map,
            shortcut_type=shortcut_type,
            no_cuda=no_cuda,
        )
    return MedicalNetResNet(block, layers, shortcut_type=shortcut_type, no_cuda=no_cuda)
