# 双流融合主干网络，实现特征拼接（Late Fusion）与分类预测
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))        #将项目根目录加入搜索范围

from configs.load_config import get_main_config
try:
    from .image_stream import ImageStream
    from .proj_stream import ProjStream
except ImportError:  # pragma: no cover - supports direct script execution
    from image_stream import ImageStream
    from proj_stream import ProjStream

config = get_main_config()
config_pretrained_2d = bool(config.get("model", {}).get("pretrained_2d", True))
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
config_dropout_rate = float(config.get("model", {}).get("dropout_rate", 0.5))

class DualStreamStripNet(nn.Module):
    """
    双流网络核心分类模型。
    融合 2D 图像特征 (128D) 与 1D 物理投影特征 (128D) 进行二分类。
    """
    def __init__(
        self, 
        pretrained_2d: bool = config_pretrained_2d, 
        feature_dim: int = config_feature_dim, 
        dropout_rate: float = config_dropout_rate
    ):
        super(DualStreamStripNet, self).__init__()
        
        # 1. 实例化两个分支网络
        self.image_stream = ImageStream(pretrained=pretrained_2d, feature_dim=feature_dim)
        self.proj_stream = ProjStream(feature_dim=feature_dim)
        
        # 2. 晚期融合分类头 (Late Fusion Classifier)
        combined_dim = feature_dim + feature_dim # 128 + 128 = 256
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config_feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(config_feature_dim, 1)
            # 注意：末尾不加 Sigmoid！保持输出 Logits，供 BCEWithLogitsLoss 使用
        )

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tensor: 形状为 (B, 3, 505, 220) 的图像 Tensor
            proj_tensor: 形状为 (B, 1, 512) 的投影 Tensor
            
        Returns:
            logits: 形状为 (B, 1) 的未归一化二分类预测 Logits
        """
        # 分别提取两个流的特征 -> 各自 (B, 128)
        img_feats = self.image_stream(img_tensor)
        proj_feats = self.proj_stream(proj_tensor)
        
        # 特征拼接 (Concat) -> (B, 256)
        fused_feats = torch.cat([img_feats, proj_feats], dim=1)
        
        # 输出 Logits -> (B, 1)
        logits = self.classifier(fused_feats)
        
        return logits

    def freeze_image_stream(self) -> None:
        """阶段一（20 epochs）：冻结 2D 图像流"""
        self.image_stream.freeze_backbone()

    def unfreeze_image_stream(self) -> None:
        """阶段二（10 epochs）：解冻 2D 图像流微调"""
        self.image_stream.unfreeze_backbone()

if __name__ == "__main__":
    try:
        from .inference_wrapper import InferenceWrapper
    except ImportError:  # pragma: no cover - supports direct script execution
        from inference_wrapper import InferenceWrapper
    # 模拟构建一个模型实例
    net = DualStreamStripNet(pretrained_2d=False)
    
    # 构造假数据 (Batch Size = 4)
    dummy_img = torch.randn(4, 3, 505, 220) # 2D 标准图像
    dummy_proj = torch.randn(4, 1, 512)    # 1D 物理重采样投影
    
    # 前向传播测试
    logits = net(dummy_img, dummy_proj)
    print("Logits Output Shape:", logits.shape) # 应输出: torch.Size([4, 1])
    
    # 推理封装测试
    infer_model = InferenceWrapper(net)
    probs = infer_model(dummy_img, dummy_proj)
    print("Probs Output Shape: ", probs.shape)  # 应输出: torch.Size([4, 1])
    print("Sample Output Prob: ", probs[0].item()) # 应输出 0~1 之间的概率值