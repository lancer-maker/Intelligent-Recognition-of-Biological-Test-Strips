# 2D 图像特征提取流，基于预训练的 EfficientNet-B0。
import os
import sys

# 先设置 Hugging Face 下载源，避免 timm 在创建模型时直接尝试访问默认官方源
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))        #将项目根目录加入搜索范围
from configs.load_config import get_main_config

config = get_main_config()
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))

PRETRAINED_WEIGHTS_DIR = PROJECT_ROOT / "data" / "pretrained_weights"


def _load_local_pretrained_weights(model: nn.Module, model_name: str) -> None:
    """优先从仓库本地目录加载预训练权重；若不存在则使用 timm 默认行为。"""
    weight_path = PRETRAINED_WEIGHTS_DIR / f"{model_name}.pth.bin"
    if not weight_path.exists():
        return

    checkpoint = torch.load(str(weight_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"预训练权重文件格式不正确: {weight_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "")
        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(cleaned_state_dict, strict=False)


class ImageStream(nn.Module):
    """
    2D 图像流特征提取网络。
    使用 ImageNet 预训练的 EfficientNet-B0 提取全局纹理与条带空间特征。
    """
    def __init__(self, pretrained: bool = True, feature_dim: int = config_feature_dim):
        super(ImageStream, self).__init__()

        # 1. 先创建骨干网络，避免 timm 直接从网络下载权重
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)

        # 2. 优先尝试加载本地仓库中的预训练权重
        if pretrained:
            _load_local_pretrained_weights(self.backbone, "efficientnet_b0")
            if not any(param.requires_grad for param in self.backbone.parameters()):
                pass

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