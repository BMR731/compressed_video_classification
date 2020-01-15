from backbone.resnet2d import *
import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.25


class MvNet(ResNet34):
    def __init__(self, num_segments):
        super(MvNet, self).__init__(input_channels=2, num_classes=101, pretrained=True)
        self.num_segments = num_segments
        self.data_bn = nn.BatchNorm2d(2)

    # forward train joint
    def forward(self, x):
        x = x.view((-1,) + x.size()[-3:])
        x = self.data_bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        conv1_feature = x
        x = self.layer1(x)
        low_level_feature = x
        x = self.layer2(x)
        layer2_feature = x
        x = self.layer3(x)
        layer3_feature = x
        x = self.layer4(x)
        layer4_feature = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)

        scores = x.view((-1, self.num_segments) + x.size()[1:])
        scores = torch.mean(scores, dim=1)
        return conv1_feature, low_level_feature, layer2_feature, layer3_feature, layer4_feature, x, scores

    # forward train alone
    def forward_alone(self,inputs):
        _,x = inputs
        x = x.view((-1,) + x.size()[-3:])
        x = self.data_bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        conv1_feature = x
        x = self.layer1(x)
        low_level_feature = x
        x = self.layer2(x)
        layer2_feature = x
        x = self.layer3(x)
        layer3_feature = x
        x = self.layer4(x)
        layer4_feature = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)

        scores = x.view((-1, self.num_segments) + x.size()[1:])
        scores = torch.mean(scores, dim=1)
        return scores

class IframeNet(ResNet152):
    def __init__(self, num_segments):
        super(IframeNet, self).__init__(input_channels=3, num_classes=101, pretrained=True)
        self.num_segments = num_segments
        self.mvnet = MvNet(self.num_segments)

        self.conv1x1_conv1_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
        self.conv1x1_conv1_spatial = nn.Conv2d(64, 1, 1, bias=True)

        self.conv1x1_layer1_channel_wise = nn.Conv2d(64 * 4, 64 * 4, 1, bias=True)
        self.conv1x1_layer1_spatial = nn.Conv2d(64, 1, 1, bias=True)

        self.conv1x1_layer2_channel_wise = nn.Conv2d(128 * 4, 128 * 4, 1, bias=True)
        self.conv1x1_layer2_spatial = nn.Conv2d(128, 1, 1, bias=True)

        self.conv1x1_layer3_channel_wise = nn.Conv2d(256 * 4, 256 * 4, 1, bias=True)
        self.conv1x1_layer3_spatial = nn.Conv2d(256, 1, 1, bias=True)

        self.conv1x1_layer4_channel_wise = nn.Conv2d(512 * 4, 512 * 4, 1, bias=True)
        self.conv1x1_layer4_spatial = nn.Conv2d(512, 1, 1, bias=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.init_conv1x1()

    # forward alone
    def forward_alone(self, inputs):
        x,_ = inputs
        x = x.view((-1,) + x.size()[-3:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        scores = x.view((-1, self.num_segments) + x.size()[1:])
        scores = torch.mean(scores, dim=1)
        return scores

    # train joint
    def forward(self, inputs):
        img, mv = inputs
        mv_conv1_feature, mv_layer1_feature, mv_layer2_feature, mv_layer3_feature, mv_layer4_feature, _, scores_mv = self.mvnet(
            mv)
        x = img.view((-1,) + img.size()[-3:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        conv1_feature = x
        # x = self.MGA_tmc(x, mv_conv1_feature, self.conv1x1_conv1_channel_wise, self.conv1x1_conv1_spatial)
        after_conv1_feature = x

        x = self.layer1(x)
        layer1_feature = x
        # x = self.MGA_tmc(x, mv_layer1_feature, self.conv1x1_layer1_channel_wise, self.conv1x1_layer1_spatial)
        after_layer1_feature = x

        x = self.layer2(x)
        layer2_feature = x
        x = self.MGA_tmc(x, mv_layer2_feature, self.conv1x1_layer2_channel_wise, self.conv1x1_layer2_spatial)
        after_layer2_feature = x

        x = self.layer3(x)
        layer3_feature = x
        x = self.MGA_tmc(x, mv_layer3_feature, self.conv1x1_layer3_channel_wise, self.conv1x1_layer3_spatial)
        after_layer3_feature = x

        x = self.layer4(x)
        layer4_feature = x
        x = self.MGA_tmc(x, mv_layer4_feature, self.conv1x1_layer4_channel_wise, self.conv1x1_layer4_spatial)
        after_layer4_feature = x

        img_feat_lst = [conv1_feature, layer1_feature, layer2_feature, layer3_feature, layer4_feature]
        img_feat_attentioned_lst = [after_conv1_feature, after_layer1_feature, after_layer2_feature,
                                    after_layer3_feature,
                                    after_layer4_feature]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        scores_iframe = x.view((-1, self.num_segments) + x.size()[1:])
        scores_iframe = torch.mean(scores_iframe, dim=1)

        # scores_iframe = nn.LogSoftmax(dim=1)(scores_iframe)
        # scores_mv = nn.LogSoftmax(dim=1)(scores_mv)
        return scores_iframe,scores_mv, img_feat_lst, img_feat_attentioned_lst

    def MGA_tmc(self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial):
        # for example
        # self.conv1x1_conv1_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
        # self.conv1x1_conv1_spatial = nn.Conv2d(64, 1, 1, bias=True)
        # spatial attention
        flow_feat_map = conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)

        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat

    def init_conv1x1(self):
        for k, v in self.state_dict().items():
            if 'conv1x1' in k:
                if 'weight' in k:
                    nn.init.kaiming_normal_(v)
                elif 'bias' in k:
                    nn.init.constant_(v, 0)
