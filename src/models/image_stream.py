# 2D 图像特征提取流，基于预训练的 EfficientNet-B0。
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))        #将项目根目录加入搜索范围
from configs.load_config import get_main_config

config = get_main_config()
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))

class ImageStream(nn.Module):
    """
    2D 图像流特征提取网络。
    使用 ImageNet 预训练的 EfficientNet-B0 提取全局纹理与条带空间特征。
    """
    def __init__(self, pretrained: bool = True, feature_dim: int = config_feature_dim):
        super(ImageStream, self).__init__()
        
        # 1. 加载 EfficientNet-B0 骨干网络 (num_classes=0 会移除原分类头，保留全局池化)
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained, 
            num_classes=0
        )
        
        # EfficientNet-B0 全局池化后的默认输出特征维度为 1280
        in_features = self.backbone.num_features # 1280
        
        # 2. 降维全连接层，将特征统一降维至 128 维
        self.fc = nn.Linear(in_features, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为 (B, 3, 505, 220) 的图像 Tensor
        Returns:
            out: 形状为 (B, 128) 的图像特征向量
        """
        # 特征提取 -> (B, 1280)
        feats = self.backbone(x)
        # 降维 + 激活 -> (B, 128)
        out = F.relu(self.fc(feats))
        return out

    def freeze_backbone(self) -> None:
        """阶段一训练使用：冻结 Backbone 所有权重"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """阶段二微调使用：解冻 Backbone 所有权重"""
        for param in self.backbone.parameters():
            param.requires_grad = True