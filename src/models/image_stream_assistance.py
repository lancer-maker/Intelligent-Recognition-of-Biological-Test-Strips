"""
=====================================================================
image_stream_assistance.py —— 图像流条带位置辅助模块
=====================================================================
目的: 强制模型(尤其 2D 图像流)学习"条带在哪里", 把注意力引导到正确区域,
      减少对试纸背景/污染的过响应。

设计 (双分支训练时的辅助监督, 不修改原 DualStreamStripNet):
  1. 主分类: DualStreamStripNet 原有 BCE 分类不变;
  2. 辅助定位: 在图像流特征 (feature_dim, 默认 128) 后接一个轻量
     StripPositionHead, 输出 (B, 2):
        - col 0: 条带存在性 logit (sigmoid -> 条带是否位于中央 ROI)
        - col 1: 条带中心 y 坐标 logit (回归, 高度方向 0~1)
  3. 弱标签 (数据无条带标注时自动生成):
        - 阳性样本: presence=1, y ≈ pos_center(0.5, 可加随机扰动模拟偏移)
        - 阴性样本: presence=0, y 不参与损失(置 0 并由 mask 忽略)
  4. 总损失 = 主分类 BCE + λ * (presence BCE + center MSE 忽略阴性)
     λ 等超参在启动文件(new_env_train_D.py)顶部"配置区"定义, 便于替换。

本模块提供 (供 new_env_train_D.py 等启动文件调用):
  - StripPositionHead      : 位置辅助头
  - ImageStreamAuxNet      : 组合模型 (复用基模型的 image/proj/classifier,
                             前向一次算出主 logits 与辅助输出, 避免重复前向)
  - make_strip_pseudo_labels: 生成 presence/y 弱标签
  - strip_position_loss    : 计算辅助损失 (presence + center)
=====================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_stream_net import DualStreamStripNet


class StripPositionHead(nn.Module):
    """条带位置辅助头: 输入图像流特征 -> (B, 2) = [条带存在性 logit, 中心 y logit]。"""

    def __init__(self, feature_dim: int = 128, hidden: int = 64):
        super(StripPositionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, img_feats: torch.Tensor) -> torch.Tensor:
        """img_feats: (B, feature_dim) -> out: (B, 2)。"""
        return self.net(img_feats)


class ImageStreamAuxNet(nn.Module):
    """在基模型(DualStreamStripNet)外附加图像流辅助头的组合模型。

    复用基模型的 image_stream / proj_stream / classifier, 一次前向同时给出
    主分类 logits 与辅助头输出, 供 D 分支训练使用。
    注意: 只把"基模型 base"保存为权重 (不含辅助头), 与 new_env_test 等加载兼容。
    """

    def __init__(self, base: DualStreamStripNet, position_head: StripPositionHead):
        super(ImageStreamAuxNet, self).__init__()
        self.base = base
        self.position_head = position_head

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor):
        """返回 (logits, aux_logits)。"""
        img_feats = self.base.image_stream(img_tensor)          # (B, feature_dim)
        proj_feats = self.base.proj_stream(proj_tensor)         # (B, feature_dim)
        logits = self.base.classifier(torch.cat([img_feats, proj_feats], dim=1))
        aux_logits = self.position_head(img_feats)              # (B, 2)
        return logits, aux_logits


def make_strip_pseudo_labels(
    labels: torch.Tensor,
    pos_center: float = 0.5,
    jitter: float = 0.0,
) -> tuple:
    """由分类弱标签生成条带位置伪标签。

    Args:
        labels: (B,) 0/1 分类标签 (float)
        pos_center: 阳性条带中心 y 的默认值 (高度方向归一化 0~1, 通常 0.5)
        jitter:    阳性伪标签的随机扰动幅度 (模拟实际条带偏移, >0 时启用)

    Returns:
        (presence, y_target): 均为 (B,)
            presence = 1 表示"条带存在于中央 ROI"; y_target 阳性=pos_center(+扰动), 阴性=0(忽略)
    """
    labels = labels.float().detach()
    presence = (labels > 0.5).float()
    if jitter > 0:
        noise = torch.randn_like(labels) * float(jitter) * presence
        y_target = (pos_center + noise).clamp(0.0, 1.0) * presence
    else:
        y_target = torch.full_like(labels, float(pos_center)) * presence
    return presence, y_target


def strip_position_loss(
    head_logits: torch.Tensor,
    presence: torch.Tensor,
    y_target: torch.Tensor,
    center_weight: float = 1.0,
) -> dict:
    """辅助定位损失。

    - presence 项: BCEWithLogits (对所有样本监督"有无条带")
    - center 项  : 仅对阳性样本 (presence=1) 计算 y 回归 MSE, 阴性忽略位置

    Returns:
        {"presence_loss": Tensor, "center_loss": Tensor, "total": Tensor}
    """
    presence = presence.float()
    y_target = y_target.float()
    presence_loss = F.binary_cross_entropy_with_logits(head_logits[:, 0], presence)
    mask = presence > 0.5
    if bool(mask.any()):
        center_loss = F.mse_loss(head_logits[:, 1][mask], y_target[mask])
    else:
        center_loss = head_logits.new_zeros(())
    total = presence_loss + center_weight * center_loss
    return {"presence_loss": presence_loss, "center_loss": center_loss, "total": total}
