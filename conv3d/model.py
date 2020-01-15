"""Model definition."""

from torch import nn
import torch.nn.functional as F
import torchvision
import torch
from train_options import parser
from torchvision.models.utils import load_state_dict_from_url
from torch.nn.modules import Conv3d
import numpy as np
from backbone.resnet3d import R2Plus1d
# from train_siamese import DEVICES
args = parser.parse_args()


# Flatten layer
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class IframeNet(R2Plus1d):
    def __init__(self):
        super(IframeNet,self).__init__(num_classes=101)
        self.mvnet = MVNet(pretrained=True)
        # self.init_conv1x1()


    # train
    def forward(self, inputs):
        x,y = inputs
        output_mv,layer1_feature,layer2_feature,layer3_feature,layer4_feature = self.mvnet(y)

        x = self.stem(x)

        x = self.layer1(x)
        x = x + x.mul(layer1_feature)
        x = self.layer2(x)
        x = x + x.mul(layer2_feature)
        x = self.layer3(x)
        x = x + x.mul(layer3_feature)
        x = self.layer4(x)
        x = x + x.mul(layer4_feature)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        output_iframe = self.fc_layer(x)
        return output_iframe,output_mv


class MVNet(R2Plus1d):
    def __init__(self,pretrained=True):
        super(MVNet,self).__init__(input_channels=2,num_classes=101)
        checkpoint = torch.load(r'r2+1d_bt_48_seg_5_conv3d-mv__best.pth.tar')
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        self.load_state_dict(base_dict)


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
        x = x.flatten(1)
        output = self.fc_layer(x)

        return output, layer1_feature, layer2_feature, layer3_feature, layer4_feature

# class Model(nn.Module):
#     def __init__(self, num_class, num_segments):
#         super(Model, self).__init__()
#         self.num_segments = num_segments
#
#         print(("""
# Initializing model:
#     num_segments:       {}.
#         """.format(self.num_segments)))
#
#         self.data_bn_channel_2 = nn.BatchNorm3d(2)  # input channel is 2
#         self.data_bn_channel_3 = nn.BatchNorm3d(3)  # input channel is 3
#         self.base_model_channel_3 =
#         self.base_model_channel_2 =
#         self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         # base_out = getattr(self.base_model_channel_3, 'fc_layer').out_features # preset 512
#         self.conv1x1_spatial = nn.Conv3d(in_channels=512, out_channels=1, kernel_size=1, bias=True)
#         self.conv1x1_channel_wise = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=1, bias=True)
#         self.fc_layer_1 = nn.Linear(512, 512)
#         self.dropout = nn.Dropout(DROPOUT)
#         self.clf_layer = nn.Linear(512, num_class)
#
#     def forward(self, inputs):
#         # (img,mv)
#         img, mv = inputs
#
#         x_img = self.data_bn_channel_3(img)
#         x_img = self.base_model_channel_3(x_img)
#         # for mv and residual need batch_normalization
#         # x_mv = self.data_bn_channel_2(mv)
#         # x_mv = self.base_model_channel_2(x_mv)
#
#         # x = self.MGA_tmc(x_img, x_mv,self.conv1x1_channel_wise,self.conv1x1_spatial)
#         x = self.avgpool(x_img)
#         x = x.flatten(1)
#         x = self.fc_layer_1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         out = self.clf_layer(x)
#         return out
#
#     def MGA_tmc_3d(self, f_img, f_mv, conv1x1_channel_wise, conv1x1_spatial):
#         # self.conv1x1_conv1_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
#         # self.conv1x1_conv1_spatial = nn.Conv2d(64, 1, 1, bias=True)
#
#         # spatial attention
#         mv_feature_map = conv1x1_spatial(f_mv)
#         mv_feature_map = nn.Sigmoid()(mv_feature_map)
#
#         spatial_attentioned_img_feat = mv_feature_map * f_img
#
#         # channel-wise attention
#         feat_vec = self.avgpool(spatial_attentioned_img_feat)
#         feat_vec = conv1x1_channel_wise(feat_vec)
#         feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
#         channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec
#
#         final_feat = channel_attentioned_img_feat + f_img
#         return final_feat
#
#     def MGA_t(self,f_img,f_mv,conv1x1_spatial):
#         mv_feature = conv1x1_spatial(f_mv)
#         final_feat = f_img + mv_feature*f_img
#         return final_feat
#
#     def _init_conv1x1_weights(self):
#         for k, v in self.state_dict().items():
#             if 'conv1x1' in k:
#                 if 'weight' in k:
#                     nn.init.kaiming_normal_(v)
#                 elif 'bias' in k:
#                     nn.init.constant_(v, 0)
#         return self
#
# #
