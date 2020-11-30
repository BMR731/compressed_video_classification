from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torch.nn import functional as F

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


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


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, output_channels, stride, conv_block=None, ):
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


class Timeception(nn.Module):
    '''
    keep temporal scale same
    '''

    def __init__(self, in_channels, out_channels, stride):
        super(Timeception, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.branch5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.conv1 = nn.Conv1d(3 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x (-1,c,t)
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x = torch.cat((x1, x3, x5), dim=1)
        x = self.conv1(x)
        return x


class Conv2Plus1DSpatioalEnhanced(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1DSpatioalEnhanced, self).__init__()
        self.inception = InceptionA(in_planes, out_planes, stride=stride, conv_block=BasicConv2d)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.temp_cov = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                                  padding=(padding, 0, 0), bias=False)

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
        return (stride, stride, stride)


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
        self.timeception = Timeception(out_planes, out_planes, stride)

    def forward(self, x):
        '''
        :param x:  (B, C, T, W, H)
        :return:
        '''
        x = self.spatioal_cov(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        b, c, t, w, h = x.shape
        x = x.permute(0, 3, 4, 1, 2)  # (None, 7, 7, 1024, 20)
        x = x.contiguous()
        x = x.view(-1, c, t)  # (None*7*7, 1024, 20)
        x = self.timeception(x)
        x = x.view(b, w, h, c, -1)  # (None, 7, 7, 1024, 20)
        x = x.permute(0, 3, 4, 1, 2)  # (None, 1024, 20, 7, 7)
        return x

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


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


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """

    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


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

    def __init__(self, pretrained=False, progress=True, block=BasicBlock,
                 conv_makers=[Conv2Plus1D] * 2 + [Conv2Plus1DSpatioalEnhanced] * 2,
                 layers=[2, 2, 2, 2],
                 stem=R2Plus1dStem, input_channels=3, num_classes=400, alpha=2,
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
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_layer = nn.Linear(512 * block.expansion, num_classes)

        # self.fusion1 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        # self.fusion2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.fusion1 = nn.Conv3d(256, out_channels=256, kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                 padding=(2, 0, 0))
        self.fusion2 = nn.Conv3d(512,512,kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0))
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

        layer3_feature, layer4_feature, feature_vector = stream
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = x + self.fusion1(layer3_feature)

        x = self.layer4(x)
        x = x + self.fusion2(layer4_feature)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc_layer(x)

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

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
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

    def __init__(self, pretrained=True, progress=True, block=BasicBlock,
                 conv_makers=[Conv2Plus1D] * 2 + [Conv2Plus1DTemporalEnhanced] * 2,
                 layers=[2, 2, 2, 2],
                 stem=R2Plus1dStem, input_channels=3, num_classes=400,
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
        self.inplanes = 64
        self.stem = stem()
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_layer = nn.Linear(512 * block.expansion, num_classes)

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

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        layer1_feature = x

        x = self.layer2(x)
        layer2_feature = x

        x = self.layer3(x)
        layer3_feature = x

        x = self.layer4(x)
        layer4_feature = x

        x = self.avgpool(x)
        # Flatten the layer to fc
        feature_vector = x.flatten(1)
        score = self.fc_layer(feature_vector)

        return layer1_feature, layer2_feature, layer3_feature, layer4_feature, feature_vector, score

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
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
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
    def __init__(self, n_classes, alpha):
        super(MSTN, self).__init__()
        self.mslt = MSLT(num_classes=n_classes, input_channels=3,alpha=alpha)
        self.lsmt = LSMT(num_classes=n_classes, input_channels=2)

    def forward(self, x):
        # x: (iframe, mv)
        iframe, mv = x
        _, _, layer3_feature, layer4_feature, feature_vector, score_mv = self.lsmt(mv)
        score_frame = self.mslt(iframe, (layer3_feature, layer4_feature, feature_vector))
        return score_mv, score_frame


if __name__ == "__main__":
    num_classes = 101
    # mv sample 2x iframe
    iframe = torch.rand(4, 3, 5, 224, 224)
    mv = torch.rand(4, 2, 10, 224, 224)
    # mslt = R2Plus1d(num_classes=101)
    # devices = [torch.device("cuda:%d" % device) for device in [0, 1]]
    # model = torch.nn.DataParallel(mslt, device_ids=[0, 1])
    # model = model.to(devices[0])
    # output = model(iframe)
    # print(output.size())
    #
    mtls = MSTN(n_classes=num_classes,alpha=2)
    devices = [torch.device("cuda:%d" % device) for device in [4, 5]]
    model = torch.nn.DataParallel(mtls, device_ids=[4, 5])
    model = model.to(devices[0])
    output = model((iframe, mv))
    print(output.size())
