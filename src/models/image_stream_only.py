"""
=====================================================================
image_stream_only.py —— 纯图像流 (去投影流) 分类模型 + 可选辅助增强头
=====================================================================
C 分支 (消融: 去掉 1D 投影流, 仅用 2D 图像流分类) 使用的模型。结构与
DualStreamStripNet 对应, 但去掉 proj_stream, 分类头只吃图像特征
(feature_dim, 默认 128), 末尾不加 Sigmoid (输出 logits 供 BCEWithLogits)。

提供 (供 new_env_train_C.py 等启动文件调用):
  - ImageOnlyBase : 主分类网络: image_stream(ImageStream) + 单流分类头
                    (Linear(feature,feature)->ReLU->Dropout->Linear(feature,1))
  - ImageOnlyAuxNet: 组合包装 (base + 可选 StripPositionHead 辅助增强头)
                     forward(img) 返回 logits; 若带辅助头则返回 (logits, aux_logits)

辅助增强模块可开关:
  - 构造 ImageOnlyAuxNet(base, position_head=None) 即关闭辅助 (纯图像分类);
  - 传入 StripPositionHead 则启用条带位置辅助监督 (弱标签/损失见
    image_stream_assistance.py), 把注意力引导到条带所在区域。

权重约定 (与 D 分支一致):
  - 主网络权重 (base) 与辅助头 (position_head) 分开保存/加载;
  - best_model.pth 保存 base (不含辅助头), position_head.pth 单独存辅助头。
=====================================================================
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))        # 将项目根目录加入搜索范围

from configs.load_config import get_main_config

try:
    from .image_stream import ImageStream
    from .image_stream_assistance import StripPositionHead
except ImportError:  # pragma: no cover - supports direct script execution
    from image_stream import ImageStream
    from image_stream_assistance import StripPositionHead

config = get_main_config()
config_pretrained_2d = bool(config.get("model", {}).get("pretrained", True))
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
config_dropout_rate = float(config.get("model", {}).get("dropout_rate", 0.5))


class ImageOnlyBase(nn.Module):
    """仅图像流分类网络 (去投影流): ImageStream 特征 -> 分类头 -> (B, 1) logits。"""

    def __init__(
        self,
        pretrained_2d: bool = config_pretrained_2d,
        feature_dim: int = config_feature_dim,
        dropout_rate: float = config_dropout_rate,
    ):
        super(ImageOnlyBase, self).__init__()
        self.image_stream = ImageStream(pretrained=pretrained_2d, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 1),
            # 末尾不加 Sigmoid, 保持 logits 供 BCEWithLogitsLoss
        )

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """img_tensor: (B, 3, 505, 220) -> logits: (B, 1)。"""
        img_feats = self.image_stream(img_tensor)          # (B, feature_dim)
        return self.classifier(img_feats)

    def freeze_image_stream(self) -> None:
        """阶段一: 冻结 2D 图像流 backbone (仅训练 fc + 分类头 + 可选辅助头)。"""
        self.image_stream.freeze_backbone()

    def unfreeze_image_stream(self) -> None:
        """阶段二: 解冻 2D 图像流 backbone 微调。"""
        self.image_stream.unfreeze_backbone()


class ImageOnlyAuxNet(nn.Module):
    """ImageOnlyBase + 可选条带位置辅助头 (可开关) 的组合模型。

    复用 base 的 image_stream / classifier, 一次前向同时给出主分类 logits
    与辅助头输出 (若 position_head 不为 None), 避免重复前向。
    注意: 只把"主网络 base"保存为权重 (不含辅助头), 与 best_model.pth 约定一致。
    """

    def __init__(self, base: ImageOnlyBase, position_head=None):
        super(ImageOnlyAuxNet, self).__init__()
        self.base = base
        self.position_head = position_head          # None = 关闭辅助增强

    def forward(self, img_tensor: torch.Tensor):
        """img_tensor: (B, 3, 505, 220)。

        Returns:
            logits: (B, 1); 若 position_head 非 None 则返回 (logits, aux_logits)
        """
        img_feats = self.base.image_stream(img_tensor)      # (B, feature_dim)
        logits = self.base.classifier(img_feats)
        if self.position_head is not None:
            return logits, self.position_head(img_feats)
        return logits


if __name__ == "__main__":
    # 模拟构建并前向测试
    net_plain = ImageOnlyBase(pretrained_2d=False)
    dummy_img = torch.randn(4, 3, 505, 220)
    print("ImageOnlyBase logits shape:", net_plain(dummy_img).shape)   # (4, 1)

    aux_net = ImageOnlyAuxNet(ImageOnlyBase(pretrained_2d=False),
                              StripPositionHead(feature_dim=config_feature_dim))
    logits, aux_logits = aux_net(dummy_img)
    print("ImageOnlyAuxNet logits shape:", logits.shape)               # (4, 1)
    print("ImageOnlyAuxNet aux shape:   ", aux_logits.shape)           # (4, 2)
