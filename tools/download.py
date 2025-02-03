"""
下载resnet18预训练模型，pdparams格式
"""
import paddle
import paddle.vision.models as models

resnet18 = models.resnet18(pretrained=True)
paddle.save(resnet18.state_dict(), 'resnet18_pretrained.pdparams')