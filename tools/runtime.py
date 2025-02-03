import paddle
import sys
sys.path.insert(0, '/project/DAMSDet')  # 替换为你项目的路径
# print(sys.path)     # 当前文件目录不等于项目目录
from ppdet.core.workspace import load_config
from ppdet.core.workspace import create

import ppdet
print(ppdet.__file__)   # 检查ppdet的路径

cfg_path = '/project/DAMSDet/configs/damsdet/MSDETR_r50vd_flir.yml'   # 替换为你测试的模型配置文件路径
cfg = load_config(cfg_path)
model = create(cfg.architecture)

inputs = {
    'vis_image': paddle.randn([1,3,640,640]),  
    'ir_image': paddle.randn([1,3,640,640]),    
    'im_shape': paddle.to_tensor([[640, 640]]),
    'scale_factor': paddle.to_tensor([[1., 1.]])
}

# 调用 FLOPs
paddle.flops(model, inputs, None, custom_ops=None, print_detail=False)  # 已修复 dynamic_flops.py bug，按官网方式不行，需要按RT-DETR的readme