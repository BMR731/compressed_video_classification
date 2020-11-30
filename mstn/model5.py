from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torch.nn import functional as F
from tensorboardX import SummaryWriter
model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}

GROUPS = 1


def calc_padding_1d(input_size, kernel_size, stride=1, dilation=1):
    """
    Calculate the padding.
    """

    # i = input
    # o = output
    # p = padding
    # k = kernel_size
    # s = stride
    # d = dilation
    # the equation is
    # o = [i + 2 * p - k - (k - 1) * (d - 1)] / s + 1
    # give that we want i = o, then we solve the equation for p gives us

    i = input_size
    s = stride
    k = kernel_size
    d = dilation

    padding = 0.5 * (k - i + s * (i - 1) + (k - 1) * (d - 1))
    padding = int(padding)

    return padding


# endregion

class Conv2Plus1D(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1DKeepTimeScaleNoDownsample(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1DKeepTimeScaleNoDownsample, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), padding=(0, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class Conv2Plus1DKeepTimeScaleDownsample(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1DKeepTimeScaleDownsample, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, groups=GROUPS, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, output_channels, stride, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        channels_chunk = int(output_channels) // 8
        self.stride = stride
        self.branch1x1 = conv_block(in_channels, channels_chunk, kernel_size=1, stride=stride)

        self.branch5x5_1 = conv_block(in_channels, channels_chunk, kernel_size=1)
        self.branch5x5_2 = conv_block(channels_chunk, channels_chunk * 4, kernel_size=5, stride=stride, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, channels_chunk, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(channels_chunk, channels_chunk * 2, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(channels_chunk * 2, channels_chunk * 2, kernel_size=3, stride=stride,
                                         padding=1)

        self.branch_pool = conv_block(in_channels, channels_chunk, kernel_size=1, stride=stride)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class DepthwiseConv1DLayer(nn.Module):
    """
    Shuffle the channels across groups.
    """

    def __init__(self, n_channels, kernel_size, padding, stride):
        super(DepthwiseConv1DLayer, self).__init__()
        self.kernel_size = kernel_size
        # TODO: support using different dilation rates.
        # padding = calc_padding_1d(n_timesteps, kernel_size)
        self.out_channels = n_channels//4
        self.bottleneck = nn.Conv3d(n_channels,self.out_channels,kernel_size=1,bias=False)
        self.depthwise_conv1d = nn.Conv1d(self.out_channels, self.out_channels, kernel_size, stride=stride,padding=padding)
        self.bn = nn.BatchNorm3d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        """
        input shape (None, 1024, 20, 7, 7), or (BN, C, T, H, W)
        """
        input = self.bottleneck(input)
        input_shape = input.size()
        n, c, t, h, w = input_shape
        # transpose and reshape to hide the spatial dimension, only expose the temporal dimension for depthwise conv
        tensor = input.permute(0, 3, 4, 1, 2)  # (None, 7, 7, 1024, 20)
        tensor = tensor.contiguous()
        tensor = tensor.view(-1, c, t)  # (None*7*7, 1024, 20)
        # depthwise conv on the temporal dimension, as if it was the spatial dimension
        tensor = self.depthwise_conv1d(tensor)  # (None*7*7, 1024, 20)
        # get timesteps after convolution
        t = tensor.size()[-1]
        # reshape to get the spatial dimensions
        tensor = tensor.view(n, h, w, c, t)  # (None, 7, 7, 1024, 20)
        # finally, transpose to get the desired output shape
        tensor = tensor.permute(0, 3, 4, 1, 2)  # (None, 1024, 20, 7, 7)
        tensor = self.bn(tensor)
        tensor = self.relu(tensor)
        return tensor


class Timeception(nn.Module):
    '''
    keep temporal scale same
    '''

    def __init__(self, in_channels):
        super(Timeception, self).__init__()
        channels = in_channels // 4

        self.branch1 = nn.Conv3d(in_channels, channels, kernel_size=1)
        self.branch2 = DepthwiseConv1DLayer(in_channels, kernel_size=3, stride=1, padding=1)
        self.branch3 = DepthwiseConv1DLayer(in_channels, kernel_size=5, stride=1, padding=2)  # 全局一点
        self.branch4 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.bn = nn.BatchNorm3d(in_channels)
        self.bottleneck = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x (-1,c,t)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x4 = self.bn(x4)
        x4 = self.bottleneck(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x

    # def forward(self, x):  # 不使用简单的相加，而是通过从全局到局部的方式，逐步级联
    #     x5 = self.branch5(x)
    #     x3 = self.branch3(x) + x5
    #     x1 = self.branch1(x) + x3
    #     return x1


class Conv2Plus1DSpatioalEnhanced(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1):
        super(Conv2Plus1DSpatioalEnhanced, self).__init__()
        self.inception = InceptionA(in_planes, out_planes, stride=stride, conv_block=BasicConv2d)
        self.bn3 = nn.BatchNorm3d(out_planes)

        self.temp_cov = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=(1, 1, 1),
                                  padding=(1, 0, 0), bias=False)  # keep time scale

    def forward(self, x):
        '''
        :param x:  (N, C, T, W, H)
        :return:
        '''
        n, c, t, w, h = x.shape
        x = x.view(-1, c, w, h)
        x = self.inception(x)
        x = x.view(n, x.shape[1], -1, x.shape[2], x.shape[3])
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.temp_cov(x)
        return x

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class Conv2Plus1DTemporalEnhanced(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1DTemporalEnhanced, self).__init__()
        self.spatioal_cov = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                                      padding=(0, padding, padding), bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.timeception = Timeception(out_planes)

    def forward(self, x):
        '''
        :param x:  (B, C, T, W, H)
        :return:
        '''
        x = self.spatioal_cov(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.timeception(x)
        return x

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SpatialStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """

    def __init__(self):
        super(SpatialStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))


class TimeStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """

    def __init__(self, alpha):
        super(TimeStem, self).__init__(
            nn.Conv3d(2, 64 // alpha, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64 // alpha),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class MSLT(nn.Module):

    def __init__(self, alpha, pretrained=False, progress=True, block=Bottleneck,
                 conv_makers=[Conv2Plus1DKeepTimeScaleDownsample] * 4,
                 layers=[3, 4, 6, 3],
                 stem=SpatialStem, input_channels=3,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(MSLT, self).__init__()
        self.alpha = alpha
        self.inplanes = 64 + 64 // self.alpha
        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        # self.pooling = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.fusion0 = nn.Conv3d(64 // self.alpha, 64 // self.alpha, kernel_size=(3, 1, 1),
                                 stride=(self.alpha, 1, 1), padding=(1, 0, 0),bias=False)
        self.fusion1 = nn.Conv3d(256 // self.alpha, 256 // self.alpha, kernel_size=(3, 1, 1),
                                 stride=(self.alpha, 1, 1), padding=(1, 0, 0),bias=False)
        self.fusion2 = nn.Conv3d(512 // self.alpha, 512 // self.alpha, kernel_size=(3, 1, 1),
                                 stride=(self.alpha, 1, 1), padding=(1, 0, 0,),bias=False)
        self.fusion3 = nn.Conv3d(1024 // self.alpha, 1024 // self.alpha, kernel_size=(3, 1, 1),
                                 stride=(self.alpha, 1, 1), padding=(1, 0, 0),bias=False)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['r2plus1d_18'],
                                                  progress=progress)

            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict, strict=False)

        if input_channels != 3:
            self.stem[0] = nn.Conv3d(input_channels, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                     bias=True)

    def forward(self, x, stream):

        stem_feature, layer1_feature, layer2_feature, layer3_feature, layer4_feature = stream
        x = self.stem(x)
        x = self.layer1(torch.cat([x, self.fusion0(stem_feature)], dim=1))
        x = self.layer2(torch.cat([x, self.fusion1(layer1_feature)], dim=1))
        x = self.layer3(torch.cat([x, self.fusion2(layer2_feature)], dim=1))
        x = self.layer4(torch.cat([x, self.fusion3(layer3_feature)], dim=1))
        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))
        self.inplanes = planes * block.expansion + planes * block.expansion // self.alpha  # TODO important
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d,nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class LSMT(nn.Module):

    def __init__(self, alpha, pretrained=False, progress=True, block=Bottleneck,
                 conv_makers=[Conv2Plus1DKeepTimeScaleDownsample] * 1 + [Conv2Plus1DTemporalEnhanced] * 3,
                 layers=[3, 4, 6, 3], input_channels=2,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(LSMT, self).__init__()
        self.alpha = alpha
        self.inplanes = 64 // self.alpha
        self.stem = TimeStem(self.alpha)

        self.layer1 = self._make_layer(block, conv_makers[0], 64 // self.alpha, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128 // self.alpha, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256 // self.alpha, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512 // self.alpha, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['r2plus1d_18'],
                                                  progress=progress)

            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(pretrained_dict, strict=False)

    def forward(self, x):
        x = self.stem(x)
        stem_feature = x
        # print(stem_feature.shape)
        x = self.layer1(x)
        layer1_feature = x
        # print(layer1_feature.shape)
        x = self.layer2(x)
        layer2_feature = x
        # print(layer2_feature.shape)
        x = self.layer3(x)
        layer3_feature = x

        x = self.layer4(x)
        layer4_feature = x

        x = self.avgpool(x)

        x = x.flatten(1)

        return (stem_feature, layer1_feature, layer2_feature, layer3_feature, layer4_feature), x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d,nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class MSTN(nn.Module):  # multi-scale spational and temporal network
    def __init__(self, num_classes, alpha, dropout=0.5):
        super(MSTN, self).__init__()
        self.mslt = MSLT(input_channels=3, alpha=alpha)
        self.lsmt = LSMT(input_channels=2, alpha=alpha)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(2048 + 2048 // alpha, num_classes)
        self.fc_slow = nn.Linear(2048,num_classes)
        self.fc_fast = nn.Linear(2049//alpha,num_classes)

    def forward(self, x):
        # x: (iframe, mv)
        iframe, mv = x
        feature_stream, flatten1 = self.lsmt(mv)
        flatten2 = self.mslt(iframe, feature_stream)
        x = torch.cat([flatten1, flatten2], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        score_fast = self.fc_fast(flatten1)
        score_slow = self.fc_slow(flatten2)
        return x, score_fast, score_slow


def model_complexity_metric():
    from ptflops import get_model_complexity_info
    def input_constructor(input_res):
        im1 = torch.randn(size=(1, 3, 1, 224, 224))
        mv1 = torch.randn(size=(1, 2, 8, 224, 224))
        return {'x': [im1, mv1]}

    net = MSTN(num_classes=400, alpha=8)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, input_res=(3, 224, 224), input_constructor=input_constructor,
                                                 as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == "__main__":
    import os

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = cur_dir + r'/log'
    with SummaryWriter(logdir=log_dir,comment='model5') as w:
        # # mv sample 2x iframe
        iframe = torch.rand(1, 3, 4, 224, 224)
        mv = torch.rand(1, 2, 16, 224, 224)
        model = MSTN(num_classes=400, alpha=4)
        w.add_graph(model,((iframe,mv),))
    # output = model((iframe, mv))
    # devices = [torch.device("cuda:%d" % device) for device in [6, 7]]
    # model = torch.nn.DataParallel(mtls, device_ids=[6, 7])
    # model = model.to(devices[0])
    # output = model((iframe, mv))
    #          print(output.size())
    print("finished")
    # model_complexity_metric()
