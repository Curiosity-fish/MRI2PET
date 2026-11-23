from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ResNetBackbone",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
]


def conv3x3x3(
    in_planes: int, out_planes: int, stride: int | tuple[int, int, int] = 1, dilation: int = 1
) -> nn.Conv3d:
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(
    x: torch.Tensor, planes: int, stride: int, no_cuda: bool = False
) -> torch.Tensor:
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = out.new_zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    )
    if not no_cuda and out.is_cuda:
        zero_pads = zero_pads.cuda()
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        shortcut_type: str = "B",
        in_channels: int = 1,
        no_cuda: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda

        # MedicalNet uses a single-channel stem by default.
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        self.out_channels = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        shortcut_type: str,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [
            block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def _build_resnet(
    block: type[BasicBlock | Bottleneck], layers: list[int], **kwargs
) -> ResNetBackbone:
    kwargs.pop("sample_input_D", None)
    kwargs.pop("sample_input_H", None)
    kwargs.pop("sample_input_W", None)
    kwargs.pop("num_seg_classes", None)
    return ResNetBackbone(block, layers, **kwargs)


def resnet10(**kwargs) -> ResNetBackbone:
    return _build_resnet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet18(**kwargs) -> ResNetBackbone:
    return _build_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNetBackbone:
    return _build_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs) -> ResNetBackbone:
    return _build_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs) -> ResNetBackbone:
    return _build_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs) -> ResNetBackbone:
    return _build_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet200(**kwargs) -> ResNetBackbone:
    return _build_resnet(Bottleneck, [3, 24, 36, 3], **kwargs)
