# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['MSDETR']

@register
class MSDETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_vis,
                 backbone_ir,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck_visir=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(MSDETR, self).__init__()
        self.backbone_vis = backbone_vis
        self.backbone_ir = backbone_ir
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck_visir = neck_visir
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone_vis
        backbone_vis = create(cfg['backbone_vis'])
        # backbone_ir
        backbone_ir = create(cfg['backbone_ir'])
        # neck
        kwargs = {'input_shape': backbone_vis.out_shape}
        neck_visir = create(cfg['neck_visir'], **kwargs) if cfg['neck_visir'] else None

        # transformer
        if neck_visir is not None:
            kwargs = {'input_shape': neck_visir.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_vis.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone_vis': backbone_vis,
            'backbone_ir': backbone_ir,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck_visir": neck_visir,
        }

    def _forward(self):
        # Backbone
        """
        self.inputs
        type: dict
        vis_img = 2, 3, 736, 736  B C H W
        ir_img = 2, 3, 736, 736   736是因为用了 multi_scale_train 原图尺寸resize为 480~800 随意一个数
        ir_img = 2, 3, 736, 736   
        """
        vis_body_feats = self.backbone_vis(self.inputs,1)   # 经过主干网络提取特征
        ir_body_feats = self.backbone_ir(self.inputs,2)

        # Neck
        if self.neck_visir is not None:
            visir_body_feats = self.neck_visir(vis_body_feats, ir_body_feats, None)   # vis_s5作为feature

        pad_mask = self.inputs.get('pad_mask', None)

        out_transformer = self.transformer(visir_body_feats, pad_mask, self.inputs) # decoder不变

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, None,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, None)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()