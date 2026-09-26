##################################################################################
##Our code is built on GeleNet. So in this code, GeleNet refers to our iHEPNet.##
##################################################################################
# Information Entropy-driven Detail-Context Interaction Module (iDCIM) consists of TopKChannelSelect and TLCSA.
# MEAtt refers to Hierarchical Edge Perception Module (HEPM).
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MobileViT import mobile_vit_x_small
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout

from typing import List, Callable
from torch import Tensor

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

# 深度可分离卷积 有relu
class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)

# 深度可分离卷积 无relu
class DSConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, bn=True, relu=False):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, bn=bn, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)



# out = channel_shuffle(out, 2)
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

# 无relu
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SpatialAttentionDS(nn.Module):
    def __init__(self, stride=1):
        super(SpatialAttentionDS, self).__init__()
        
        self.conv1 = DSConv(2, 1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# MEAtt refers to Hierarchical Edge Perception Module (HEPM).
# multi-edge-aware attention module, MEAtt
class MEAtt3(nn.Module):
    def __init__(self, channel):
        super(MEAtt3, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.SAEdge = SpatialAttentionDS()
        self.SA = SpatialAttentionDS()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        self.conv1 = DSConv3x3(channel, channel, stride=1)

    def forward(self, x3, x4):
        edge1 = (x3-self.upsample2(x4)).abs()
        edge_att = self.SAEdge(edge1)
        att = self.SA(x3)

        att = att + self.sigmoid(self.gamma)*edge_att
        x3 =  self.conv1(att*x3 + x3)

        return x3

# MEAtt refers to Hierarchical Edge Perception Module (HEPM).
class MEAtt2(nn.Module):
    def __init__(self, channel):
        super(MEAtt2, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.SAEdge = SpatialAttentionDS()
        self.SA = SpatialAttentionDS()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        self.conv1 = DSConv3x3(channel, channel, stride=1)

    def forward(self, x2, x3, x4):
        edge1 = (x2-self.upsample2(x3)).abs()
        edge2 = (x2-self.upsample4(x4)).abs()

        edge_att = self.SAEdge(torch.cat((edge1, edge2), 1))
        att = self.SA(x2)

        att = att + self.sigmoid(self.gamma)*edge_att
        x2 =  self.conv1(att*x2 + x2)

        return x2

# MEAtt refers to Hierarchical Edge Perception Module (HEPM).
class MEAtt1(nn.Module):
    def __init__(self, channel):
        super(MEAtt1, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.SAEdge = SpatialAttentionDS()
        self.SA = SpatialAttentionDS()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
        self.conv1 = DSConv3x3(channel, channel, stride=1)

    def forward(self, x1, x2, x3, x4):
        edge1 = (x1-self.upsample2(x2)).abs()
        edge2 = (x1-self.upsample4(x3)).abs()
        edge3 = (x1-self.upsample8(x4)).abs()

        edge_att = self.SAEdge(torch.cat((edge1, edge2, edge3), 1))
        att = self.SA(x1)

        att = att + self.sigmoid(self.gamma)*edge_att
        x1 =  self.conv1(att*x1 + x1)

        return x1

# Information Entropy-driven Detail-Context Interaction Module (iDCIM) consists of TopKChannelSelect and TLCSA.
class TopKChannelSelect(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TopKChannelSelect, self).__init__()
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.GAP = nn.AdaptiveAvgPool2d(1)

        # self.project = DSConv3x3(out_channel, out_channel, stride=1)  # 16x88x88->16x88x88

    def forward(self, x):
        n, c, h, w = x.shape

        x = x.abs()  # (N, C, H, W)

        prob = x.view(n, c, -1)  # (N, C, H*W)
        prob = prob / (prob.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化到和为1

        eps = 1e-8
        entropy = -torch.sum(prob * torch.log(prob + eps), dim=-1, keepdim=True)  # (N, C, 1)

        entropy = entropy.squeeze(-1)  # (N, C)
        _, entropy_sorted_indices = torch.sort(entropy, dim=1, descending=True)  # (n, c), 按绝对值从大到小排

        c_keep = int(c * 0.5)

        entropy_topk_indices = entropy_sorted_indices[:, :c_keep]  # (n, c_keep)
        entropy_lastk_indices = entropy_sorted_indices[:, c_keep:]  # (n, c_keep)

        entropy_topk_indices_expanded = entropy_topk_indices.unsqueeze(-1).unsqueeze(-1).expand(n, c_keep, h,
                                                                                            w)  # (n, c_keep, h, w)
        entropy_lastk_indices_expanded = entropy_lastk_indices.unsqueeze(-1).unsqueeze(-1).expand(n, c_keep, h,
                                                                                            w)  # (n, c_keep, h, w)
        topk_selected = torch.gather(x, dim=1, index=entropy_topk_indices_expanded)
        lastk_selected = torch.gather(x, dim=1, index=entropy_lastk_indices_expanded)

        return topk_selected, lastk_selected

# Information Entropy-driven Detail-Context Interaction Module (iDCIM) consists of TopKChannelSelect and TLCSA.
# Channel-wise top-last self-att
class TLCSA(nn.Module):
    def __init__(self, channel=16):
        super(TLCSA, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.to_hidden1 = nn.Conv2d(channel, channel * 3, kernel_size=1)
        self.to_hidden2 = nn.Conv2d(channel, channel * 3, kernel_size=1)

        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.conv_out = DSConv3x3(2 * channel, 2 * channel, stride=1)


    def forward(self, x1, x2): # x1_TopK, x1_LastK   x1: Q, 纹理, x2: K，平滑
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: C X H x W
        """
        m_batchsize, C, height, width = x1.size()

        q1, k1, v1 = self.to_hidden1(x1).chunk(3, dim=1)
        q2, k2, v2 = self.to_hidden2(x2).chunk(3, dim=1)

        proj_q2 = torch.transpose(q2, 2, 3).contiguous()  # B X C X W2 x H2 (W=H)
        att1 = torch.matmul(proj_q2, k1).contiguous()     # C X (W2 x H2) x (H1 X W1) = C X W2 x W1
        att1 = F.softmax(att1, dim=2)
        out1 = torch.matmul(v1, att1).contiguous()        # C X (H1 x W1) x (W2 x W1)= C X H1 X W1
        out_cur1 = self.gamma_1 * out1 + x1

        proj_q1 = torch.transpose(q1, 2, 3).contiguous()  # B X C X W1 x H1 (W=H)
        att2 = torch.matmul(proj_q1, k2).contiguous()     # C X (W1 x H1) x (H2 X W2) = C X W1 x W2
        att2 = F.softmax(att2, dim=2)
        out2 = torch.matmul(v2, att2).contiguous()        # C X (H2 x W2) x (W1 x W2)= C X H2 X W2
        out_cur2 = self.gamma_2 * out2 + x2

        out = self.conv_out(torch.cat((out_cur1, out_cur2), 1)) # 32x88x88

        return out


class PDecoder(nn.Module):
    def __init__(self, channel):
        super(PDecoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_upsample1 = DSConv(channel, channel, stride=1)
        self.conv_upsample2 = DSConv(channel, channel, stride=1)
        self.conv_upsample3 = DSConv(channel, channel, stride=1)
        self.conv_upsample4 = DSConv(channel, channel, stride=1)
        self.conv_upsample5 = DSConv(channel, channel, stride=1)
        self.conv_upsample6 = DSConv(channel, channel, stride=1)
        self.conv_upsample7 = DSConv(channel, channel, stride=1)
        self.conv_upsample8 = DSConv(2*channel, 2*channel, stride=1)
        self.conv_upsample9 = DSConv(3*channel, 3*channel, stride=1)

        self.conv_concat2 = DSConv(2*channel, 2*channel, stride=1)
        self.conv_concat3 = DSConv(3*channel, 3*channel, stride=1)
        self.conv_concat4 = DSConv(4*channel, 4*channel, stride=1)
        self.conv4 = DSConv(4*channel, channel, stride=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3, x4): # x1: 32x11x1, x2: 32x22x22, x3: 32x44x44, x4: 32x88x88,
        x1_1 = x1 # 32x11x11
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 # 32x22x22
        x3_1 = self.conv_upsample2(self.upsample4(x1)) * self.conv_upsample3(self.upsample(x2)) * x3 # 32x44x44
        x4_1 = self.conv_upsample4(self.upsample8(x1)) * self.conv_upsample5(self.upsample4(x2)) * \
               self.conv_upsample6(self.upsample(x3)) * x4# 32x88x88

        x2_2 = torch.cat((x2_1, self.conv_upsample7(self.upsample(x1_1))), 1) # 32x22x22
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample8(self.upsample(x2_2))), 1) # 32x44x44
        x3_2 = self.conv_concat3(x3_2)

        x4_2 = torch.cat((x4_1, self.conv_upsample9(self.upsample(x3_2))), 1) # 32x88x88
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x) # 1x88x88

        return x


class GeleNet(nn.Module):
    def __init__(self, channel=32):
        super(GeleNet, self).__init__()

        # self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        # mobile_vit_small()返回的特征是[64, 96, 128, 160]
        # mobile_vit_x_small()返回的特征是[48, 64, 80, 96]
        # mobile_vit_xx_small()返回的特征是[24, 48, 64, 80]
        self.backbone = mobile_vit_x_small() # [48, 64, 80, 96]

        path = './model/mobilevit_xs.pt'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = DSConv3x3(48, channel, stride=1)  # 48x88x88->32x88x88
        self.ChannelNormalization_2 = DSConv3x3(64, channel, stride=1) # 64x44x44->32x44x44
        self.ChannelNormalization_3 = DSConv3x3(80, channel, stride=1) # 80x22x22->32x22x22
        self.ChannelNormalization_4 = DSConv3x3(96, channel, stride=1) # 96x11x11->32x11x11

        # Information Entropy-driven Detail-Context Interaction Module (iDCIM) consists of TopKChannelSelect and TLCSA.
        self.TopKChannelSelect_1 = TopKChannelSelect(channel, channel // 2)  # 32x88x88->16x88x88
        self.TopKChannelSelect_2 = TopKChannelSelect(channel, channel // 2)  # 32x44x44->16x44x44
        self.TopKChannelSelect_3 = TopKChannelSelect(channel, channel // 2)  # 32x22x22->16x22x22
        self.TopKChannelSelect_4 = TopKChannelSelect(channel, channel // 2)  # 32x11x11->16x11x11

        self.TLSA_1 = TLCSA(channel // 2)
        self.TLSA_2 = TLCSA(channel // 2)
        self.TLSA_3 = TLCSA(channel // 2)
        self.TLSA_4 = TLCSA(channel // 2)

        # MEAtt refers to Hierarchical Edge Perception Module (HEPM).
        self.MEAtt1 = MEAtt1(channel)
        self.MEAtt2 = MEAtt2(channel)
        self.MEAtt3 = MEAtt3(channel)

        self.PDecoder = PDecoder(channel)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        # backbone
        MobileViT = self.backbone(x)
        x1 = MobileViT[1] # 48x88x88
        x2 = MobileViT[2] # 64x44x44
        x3 = MobileViT[3] # 80x22x22
        x4 = MobileViT[4] # 96x11x11

        x1_nor = self.ChannelNormalization_1(x1) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2) # 32x44x44
        x3_nor = self.ChannelNormalization_3(x3) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4) # 32x11x11

        # Information Entropy-driven Detail-Context Interaction Module (iDCIM) consists of TopKChannelSelect and TLCSA.
        x1_TopK, x1_LastK = self.TopKChannelSelect_1(x1_nor)  # 16x88x88
        x2_TopK, x2_LastK = self.TopKChannelSelect_2(x2_nor)  # 16x44x44
        x3_TopK, x3_LastK = self.TopKChannelSelect_3(x3_nor)  # 16x22x22
        x4_TopK, x4_LastK = self.TopKChannelSelect_4(x4_nor)  # 16x11x11

        x1_TL = self.TLSA_1(x1_TopK, x1_LastK)
        x2_TL = self.TLSA_2(x2_TopK, x2_LastK)
        x3_TL = self.TLSA_3(x3_TopK, x3_LastK)
        x4_TL = self.TLSA_4(x4_TopK, x4_LastK)

        # MEAtt refers to Hierarchical Edge Perception Module (HEPM).
        x3_MEAtt = self.MEAtt3(x3_TL, x4_TL)
        x2_MEAtt = self.MEAtt2(x2_TL, x3_TL, x4_TL)
        x1_MEAtt = self.MEAtt1(x1_TL, x2_TL, x3_TL, x4_TL)

        s1 = self.upsample4(self.PDecoder(x4_TL, x3_MEAtt, x2_MEAtt, x1_MEAtt))


        return s1



