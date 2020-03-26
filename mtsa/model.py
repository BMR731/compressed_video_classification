import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['resnet50', 'resnet101', 'resnet152', 'resnet200']


# (N,C,D,H,W)
class SETLayer(nn.Module):
    def __init__(self, depth, reduction=2):
        super(SETLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(depth, depth // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(depth // reduction, depth, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, w, h = x.size()
        # time scale reserved
        y = torch.transpose(x, 1, 2)
        y = self.avg_pool(y).view(b, d)
        y = self.fc(y).view(b, d, 1, 1, 1)
        y = torch.transpose(x, 1, 2)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


TIME_SCALE = [4, 8, 16]


class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=10, dropout=0.5):
        super(SlowFast, self).__init__()

        ## timescale-1
        self.fast1_inplanes = 16
        self.fast1_conv1 = nn.Conv3d(2, 16, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast1_bn1 = nn.BatchNorm3d(16)
        self.fast1_relu = nn.ReLU(inplace=True)
        self.fast1_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast1_res2 = self._make_layer_fast1(block, 16, layers[0], head_conv=3)
        self.fast1_res3 = self._make_layer_fast1(
            block, 32, layers[1], stride=2, head_conv=3)
        self.fast1_res4 = self._make_layer_fast1(
            block, 64, layers[2], stride=2, head_conv=3)
        self.fast1_res5 = self._make_layer_fast1(
            block, 128, layers[3], stride=2, head_conv=3)

        # timescale-2
        self.fast2_inplanes = 8
        self.fast2_conv1 = nn.Conv3d(2, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast2_bn1 = nn.BatchNorm3d(8)
        self.fast2_relu = nn.ReLU(inplace=True)
        self.fast2_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast2_res2 = self._make_layer_fast2(block, 8, layers[0], head_conv=3)
        self.fast2_res3 = self._make_layer_fast2(
            block, 16, layers[1], stride=2, head_conv=3)
        self.fast2_res4 = self._make_layer_fast2(
            block, 32, layers[2], stride=2, head_conv=3)
        self.fast2_res5 = self._make_layer_fast2(
            block, 64, layers[3], stride=2, head_conv=3)

        # timescale-3
        self.fast3_inplanes = 4
        self.fast3_conv1 = nn.Conv3d(2, 4, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast3_bn1 = nn.BatchNorm3d(4)
        self.fast3_relu = nn.ReLU(inplace=True)
        self.fast3_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast3_res2 = self._make_layer_fast3(block, 4, layers[0], head_conv=3)
        self.fast3_res3 = self._make_layer_fast3(
            block, 8, layers[1], stride=2, head_conv=3)
        self.fast3_res4 = self._make_layer_fast3(
            block, 16, layers[2], stride=2, head_conv=3)
        self.fast3_res5 = self._make_layer_fast3(
            block, 32, layers[3], stride=2, head_conv=3)



        self.slow_inplanes = 64 + 64 // 8 * 2
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(
            block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(
            block, 512, layers[3], stride=2, head_conv=3)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(4 + 8 + 16 + 2048, class_num, bias=False)

    def forward(self, input):
        iframe, mv1, mv2, mv3 = input
        fast1, lateral1 = self.FastPath1(mv1)
        fast2, lateral2 = self.FastPath2(mv2)
        fast3, lateral3 = self.FastPath3(mv3)

        attentioned_laterals = []
        for i in range(4):
            attentioned = self.timescale_attention([lateral1[i], lateral2[i], lateral3[i]])
            attentioned_laterals.append(attentioned)

        slow = self.SlowPath(iframe, attentioned_laterals)
        x = torch.cat([slow, fast1, fast2, fast3], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

    def SlowPath(self, input, lateral):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, lateral[0]], dim=1)
        x = self.slow_res2(x)
        x = torch.cat([x, lateral[1]], dim=1)
        x = self.slow_res3(x)
        x = torch.cat([x, lateral[2]], dim=1)
        x = self.slow_res4(x)
        x = torch.cat([x, lateral[3]], dim=1)
        x = self.slow_res5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def FastPath1(self, input):
        lateral = []
        x = self.fast1_conv1(input)
        x = self.fast1_bn1(x)
        x = self.fast1_relu(x)
        pool1 = self.fast1_maxpool(x)
        lateral.append(pool1)

        res2 = self.fast1_res2(pool1)
        lateral.append(res2)

        res3 = self.fast1_res3(res2)
        lateral.append(res3)

        res4 = self.fast1_res4(res3)
        lateral.append(res4)

        res5 = self.fast1_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))

        return x, lateral

    def FastPath2(self, input):
        lateral = []
        x = self.fast2_conv1(input)
        x = self.fast2_bn1(x)
        x = self.fast2_relu(x)
        pool1 = self.fast2_maxpool(x)
        lateral.append(pool1)

        res2 = self.fast2_res2(pool1)
        lateral.append(res2)

        res3 = self.fast2_res3(res2)
        lateral.append(res3)

        res4 = self.fast2_res4(res3)
        lateral.append(res4)

        res5 = self.fast2_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))

        return x, lateral

    def FastPath3(self, input):
        lateral = []
        x = self.fast3_conv1(input)
        x = self.fast3_bn1(x)
        x = self.fast3_relu(x)
        pool1 = self.fast3_maxpool(x)
        lateral.append(pool1)

        res2 = self.fast3_res2(pool1)
        lateral.append(res2)

        res3 = self.fast3_res3(res2)
        lateral.append(res3)

        res4 = self.fast3_res4(res3)
        lateral.append(res4)

        res5 = self.fast3_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))

        return x, lateral

    def timescale_attention(self, laterals):
        pool1 = nn.MaxPool3d(kernel_size=(TIME_SCALE[0], 1, 1), stride=(TIME_SCALE[0], 1, 1))
        pool2 = nn.MaxPool3d(kernel_size=(TIME_SCALE[1], 1, 1), stride=(TIME_SCALE[1], 1, 1))
        pool3 = nn.MaxPool3d(kernel_size=(TIME_SCALE[2], 1, 1), stride=(TIME_SCALE[2], 1, 1))

        x1 = pool1(laterals[0])
        x2 = pool2(laterals[1])
        x3 = pool3(laterals[2])

        channel = x1.shape[1]+x2.shape[1]+x3.shape[1]
        ##todo make sure the dim
        x = torch.cat([x1,x2,x3],dim=1)
        x = nn.Conv3d(channel,channel,1,1,0)
        x = SETLayer(x.shape[2])
        return x


    def _make_layer_fast1(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast1_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast1_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast1_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast1_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast1_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_fast2(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast2_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast2_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast2_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast2_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast2_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_fast3(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast3_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast3_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast3_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast3_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast3_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)
    
    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))

        self.slow_inplanes = planes * block.expansion + planes * block.expansion // 8 * 2
        return nn.Sequential(*layers)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == "__main__":
    num_classes = 101
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 64, 224, 224))
    mv1= torch.rand(1,2,16,224,224)
    mv2= torch.rand(1,2,32,224,224)
    mv3= torch.rand(1,2,63,224,224)
    iframe= torch.rand(1,3,4,224,224)
    input = (iframe,mv1,mv2,mv3)
    model = resnet50(class_num=num_classes)
    output = model(input)
    print(output.size())
