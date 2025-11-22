import torch
import torch.nn as nn
from network.models import get_convnext


import torch_dct as DCT
import numpy as np
import math
import torch.nn.functional as F



class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class GuidedAttention(nn.Module):
    """ Reconstruction Guided Attention. """

    def __init__(self, dim=728, drop_rate=0.2):
        super(GuidedAttention, self).__init__()

        self.gated = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            # nn.Conv2d(128, 1, 1, bias=False),
            # nn.Sigmoid()
        )

        self.att = cross_attention(dim = dim*2, num_heads=8)


        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding):
        residual_full = torch.abs(x - pred_x)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        res_map = self.gated(residual_x)

        res_map_en = self.att(embedding, res_map)

        return res_map_en + self.dropout(embedding)



class GuidedAttention_1(nn.Module):
    """ Reconstruction Guided Attention. """

    def __init__(self, dim=728, drop_rate=0.2):
        super(GuidedAttention_1, self).__init__()

        self.gated = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            # nn.Conv2d(128, 1, 1, bias=False),
            # nn.Sigmoid()
        )

        self.att = cross_attention(dim = dim*2, num_heads=8)


        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding):
        residual_full = torch.abs(x - pred_x)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        res_map = self.gated(residual_x)

        res_map_en = self.att(embedding, res_map)

        return res_map_en + self.dropout(embedding)
    






### 图像特征进行全局DCT变换, use filter 获取不同频带的系数
class Filter(nn.Module):
    # def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):   
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.






### --- obtain DCT feature based global feature
class dct_trans_global(nn.Module):
    def __init__(self, size = 7):
        super(dct_trans_global, self).__init__()
        self.mh_filter = Filter(size, size // 2.82, size * 2)   # 中高频的频带

    
    def forward(self, x):
        '''
        x: imgs (Batch, H, W, Channel)
        '''
        data = x
        num_batchsize = data.shape[0]
        channel = data.shape[1]
        size = data.shape[2]
        
        image = DCT.dct_2d(data, norm='ortho')   # 原始特征进行DCT变换

        image = self.mh_filter(image)   ### 滤波之后的DCT系数

        input_dct = DCT.idct_2d(image, norm='ortho')   # obtain middle and high feature


        return input_dct





class Convnext_H(nn.Module):
    """ Convnext with Hierarchical feature """

    def __init__(self, num_classes=2):
        super(Convnext_H, self).__init__()
        self.name = 'convnext_base'
        self.net = get_convnext(model_name=self.name, num_classes=num_classes, pretrained=True)
        # print(self.net)

        self.stem = self.net.stem   # 
        self.stage0 = self.net.stages[0]
        self.stage1 = self.net.stages[1]
        self.stage2 = self.net.stages[2]
        self.stage3 = self.net.stages[3]
        self.norm_pre = self.net.norm_pre
        self.head = self.net.head
        
        self.recon1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.recon2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.img_att1 = GuidedAttention(dim=128, drop_rate=0.2)

        self.img_att2 = GuidedAttention_1(dim=256, drop_rate=0.2)


        self.att_global = dct_trans_global()   # 获取频域信息

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.Linear = nn.Linear(in_features=1024, out_features=128)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage0(out)
        out_0 = out 

        out = self.stage1(out)
        out_1 = out

        out_re1 = self.recon1(out_1)
        out = self.img_att1(out_0, out_re1, out_1)  # 256*28*28
        out = self.stage2(out)
        out_2 = out
        out_re2 = self.recon2(out_2)
        out = self.img_att2(out_1, out_re2, out_2)  # B C 28 28
        out = self.stage3(out)
        out = self.norm_pre(out)

        fre = self.att_global(out)   # 获取特征的中高频，并逆DCT变换
        fre = self.avgpool(fre)
        fre = fre.squeeze()
        fre = self.Linear(fre)
        fre = torch.nn.functional.normalize(fre, p=2.0, dim=1, eps=1e-12, out=None)

        out = self.head(out)
        return out, fre
    


if __name__ == '__main__':
     
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dummy = torch.rand((2, 3, 224, 224)).to(device)
    size = dummy.shape[-2:]   # size:[256, 256]
    model_two = Convnext_H(num_classes=2).to(device)
    # print(model_two.extract_textures)
    # input()
    vit_out = model_two(dummy)   # RGB经对比的特征, vit_out的预测值

    print(vit_out)




