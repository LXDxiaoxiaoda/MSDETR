# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder
from ..initializer import xavier_uniform_, linear_init_
from ..layers import MultiHeadAttention
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from functools import reduce

__all__ = ['HybridEncoder','HybridEncoder_2','Trans_SK']


class CSPRepLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)

class FusionVisIR(nn.Layer):  # PaddlePaddle 中使用 nn.Layer 而不是 nn.Module
    def __init__(self, channel, reduction=4):
        super(FusionVisIR, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)  # PaddlePaddle 中使用 AdaptiveAvgPool2D
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.Conv1x1 = nn.Conv2D(2 * channel, channel, kernel_size=1)  # PaddlePaddle 中使用 Conv2D

    def forward(self, vis, ir):
        b, c, _, _ = vis.shape
        fea = vis + ir
        _vis = self.fc(self.avg_pool(vis).reshape([b, c])).reshape([b, c, 1, 1])  # 使用 reshape 而不是 view
        _ir = self.fc(self.avg_pool(ir).reshape([b, c])).reshape([b, c, 1, 1])
        fea = fea * (self.Conv1x1(paddle.concat([_vis, _ir], axis=1)))  # PaddlePaddle 中使用 paddle.concat
        return fea

@register
class TransformerLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


@register
@serializable
class HybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))
        # encoder transformer
        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, feats, for_mot=False):
        # forward函数中的注释以RT-DETR结构为例，其余结构和注释不一定相同
        assert len(feats) == len(self.in_channels)  # in_channels [512, 1024, 2048]
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)] # 统一channel 256   proj_feats = [256, 256, 256] 三个特征层

        # AIFI s5进行编码
        # encoder
        if self.num_encoder_layers > 0: # num_encoder_layers = 1
            for i, enc_ind in enumerate(self.use_encoder_idx):  # use_encoder_idx = [2] s5特征
                h, w = proj_feats[enc_ind].shape[2:]    # h w 23 23
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).transpose(
                    [0, 2, 1])  # 2 256 23 23 -> 2 256 529 -> 2 529 256
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)  # 2 529 256
                proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
                    [-1, self.hidden_dim, h, w])    # 2 529 256 -> 2 256 529 -> 2 256 23 23 f5

        # CCFM s3 s4 f5(s5经过AIFI) 进行FPN + PAN
        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat(
                    [upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]
    
@register
@serializable
class MS_HybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(MS_HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))
        # encoder transformer
        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))
            
        # Fusion module
        self.fusion_block = FusionVisIR(256)    # 放self.input_proj后面，此输出channel为256

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, feats_vis, feats_ir, for_mot=False):
        # forward函数中的注释以RT-DETR结构为例
        assert len(feats_vis) == len(self.in_channels)  # in_channels [512, 1024, 2048]
        # get projection features
        proj_feats_vis = [self.input_proj[i](feat) for i, feat in enumerate(feats_vis)] # 统一channel 256   proj_feats = [256, 256, 256] 三个特征层
        proj_feats_ir = [self.input_proj[i](feat) for i, feat in enumerate(feats_ir)] # 统一channel 256   proj_feats = [256, 256, 256] 三个特征层
        
        proj_feats = []
        # 这里可以对特征融合
        for i in range(len(proj_feats_vis)):
            # proj_feats.append(self.fusion_block(proj_feats_vis[i], proj_feats_ir[i]))
            proj_feats.append(proj_feats_vis[i] + proj_feats_ir[i])     # 默认是简单相加


        # AIFI s5进行编码
        # encoder
        if self.num_encoder_layers > 0: # num_encoder_layers = 1
            for i, enc_ind in enumerate(self.use_encoder_idx):  # use_encoder_idx = [2] s5特征
                h, w = proj_feats[enc_ind].shape[2:]    # h w 23 23
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).transpose(
                    [0, 2, 1])  # 2 256 23 23 -> 2 256 529 -> 2 529 256
                
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)  # 2 529 256
                proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
                    [-1, self.hidden_dim, h, w])    # 2 529 256 -> 2 256 529 -> 2 256 23 23 f5

        # CCFM s3 s4 f5(s5经过AIFI) 进行FPN + PAN
        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat(
                    [upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]

@register
class Sknet2(nn.Layer):
    def __init__(self,c1):
        super(Sknet2,self).__init__()
        self.C1 = c1
        self.pool = nn.AdaptiveAvgPool2D(1)

        temp = int((c1/4)*2)
        self.fc1 = nn.Conv2D(c1, temp, 1, bias_attr=False)
        self.lk = nn.LeakyReLU()
        self.fc2 = nn.Conv2D(temp, c1*2, 1, 1, bias_attr=False)

    def forward(self,mm):
        batch_size = mm[0].shape[0]
        U = reduce(lambda x, y: x + y,mm)
        a_b = self.pool(U)
        a_b = self.fc1(a_b)
        a_b = self.lk(a_b)
        a_b = self.fc2(a_b)
        a_b = paddle.reshape(a_b,[batch_size, 2, self.C1, -1])
        a_b = nn.Softmax(axis=1)(a_b)
        a_b = list(a_b.chunk(2, axis=1))

        a_b = list(map(lambda x:paddle.reshape(x,[batch_size, self.C1, 1, 1]), a_b))
        V = list(map(lambda x, y: x*y,mm,a_b))
        V = reduce(lambda x, y:x + y, V)

        return V

@register
@serializable
class Trans_SK(nn.Layer):
    __inject__ = ['encoder_layer']
    def __init__(self,test,
                 encoder_layer='TransformerLayer'):
        super(Trans_SK,self).__init__()

        self.encoder_trans = TransformerEncoder(encoder_layer, 1)
        self.Sk1 = Sknet2(256)
        self.Sk2 = Sknet2(256)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self,vis_body_feats,ir_body_feats ):
        vis_body_feat3 = vis_body_feats[2]
        ir_body_feat3 = ir_body_feats[2]

        add_body_feat3 = vis_body_feat3 + ir_body_feat3

        h, w = add_body_feat3.shape[2:]
        # flatten [B, C, H, W] to [B, HxW, C]
        src_flatten = add_body_feat3.flatten(2).transpose(
            [0, 2, 1])

        pos_embed = self.build_2d_sincos_position_embedding(
            w, h, 256, 10000)
        memory = self.encoder_trans(src_flatten, pos_embed=pos_embed)
        add_body_feat3 = memory.transpose([0, 2, 1]).reshape(
            [-1, 256, h, w])

        body_feat2 = [vis_body_feats[1], ir_body_feats[1]]
        body_feat1 = [vis_body_feats[0], ir_body_feats[0]]
        sk_body_feat2 = self.Sk2(body_feat2)
        sk_body_feat1 = self.Sk1(body_feat1)
        body_feats = [sk_body_feat1, sk_body_feat2, add_body_feat3]
        return body_feats


@register
@serializable
class HybridEncoder_2(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[128, 256, 512],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder_2, self).__init__()
        self.in_channels = [128, 256, 512]
        in_channels = [128, 256, 512]
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))
        # encoder transformer
        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).transpose(
                    [0, 2, 1])
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
                    [-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat(
                    [upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]