"""
=====================================================================
gated_fusion.py —— 跨流门控融合双流网络 (E 模式)
=====================================================================
在现有 D 模式 (双流 + 辅助定位任务) 基础上, 将"简单拼接融合"替换为
**跨流门控融合 (Cross-Stream Gated Fusion)**, 供 new_env_train_E.py 使用:

  - 图像流特征 img_feat (feature_dim, 默认 128) 保持不变;
  - 投影流特征 proj_feat (feature_dim) 经轻量门控网络 GateMLP
    输出同维度调制向量 gate (Sigmoid, 值域 0~1);
  - 对图像流特征逐元素调制: img_feat_gated = img_feat * gate;
  - 拼接保留投影流完整信息: fused = cat([img_feat_gated, proj_feat])
    -> 送入原有分类头 (Linear 2*fd -> fd -> ReLU -> Dropout -> fd -> 1)。
  - 辅助定位头 (StripPositionHead) 仍接在原始 img_feat 上, 不经过门控
    (避免干扰条带定位任务)。

与 D 的关系:
  - 主分类/辅助定位范式与 D 完全一致 (复用同一辅助头与伪标签/损失),
    唯一差异 = 融合方式 (cat 直接拼接 vs 门控调制后拼接)。

设计要点 (D 兼容):
  - GatedDualStreamNet 与 DualStreamStripNet 保持相同外部接口
    (forward(img, proj) -> logits; .image_stream / .proj_stream /
     freeze_image_stream / unfreeze_image_stream), 因此可复用
    helper 的 validate / build_stage2_objects 与逐层解冻工具;
  - 不修改 dual_stream_net.py / image_stream_assistance.py 等既有代码,
    不影响已训练模型加载兼容。

提供 (供 new_env_train_E.py 等启动文件调用):
  - GateMLP              : 轻量门控网络 (fd -> hidden -> fd, ReLU 后 Sigmoid)
  - GatedDualStreamNet   : 门控融合主网络 (分类, 不含辅助头)
  - GatedDualStreamAuxNet: 组合包装 (base + StripPositionHead), 主 logits 走门控
                           融合, 辅助头接原始 img_feat
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
    from .proj_stream import ProjStream
    from .image_stream_assistance import StripPositionHead
except ImportError:  # pragma: no cover - supports direct script execution
    from image_stream import ImageStream
    from proj_stream import ProjStream
    from image_stream_assistance import StripPositionHead

config = get_main_config()
config_pretrained_2d = bool(config.get("model", {}).get("pretrained", True))
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
config_dropout_rate = float(config.get("model", {}).get("dropout_rate", 0.5))


class GateMLP(nn.Module):
    """轻量门控网络: proj_feat (feature_dim) -> gate (feature_dim, Sigmoid 0~1)。

    结构: Linear(fd, hidden) -> ReLU -> Linear(hidden, fd) -> Sigmoid。
    初始化: 第二层 Linear 偏置设为正值 (1.0), 使初始 gate 偏大 (接近 1),
    即门控开始时几乎不起作用, 保持与 D 模式(直接拼接)接近, 避免训练初期不稳定。
    """

    def __init__(self, feature_dim: int = config_feature_dim, hidden: int = 64):
        super(GateMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid(),
        )
        # 最后一层 Linear (index 2) bias = 1.0 => 初始 sigmoid(logit≈1) 较大, 接近不门控
        nn.init.constant_(self.net[2].bias, 1.0)

    def forward(self, proj_feats: torch.Tensor) -> torch.Tensor:
        """proj_feats: (B, feature_dim) -> gate: (B, feature_dim), 值域 (0,1)。"""
        return self.net(proj_feats)


class GatedDualStreamNet(nn.Module):
    """跨流门控融合双流网络 (E 模式核心, 不含辅助头)。

    融合流程:
        img_feat  = image_stream(img)                (B, fd)
        proj_feat = proj_stream(proj)                (B, fd)
        gate      = GateMLP(proj_feat)               (B, fd), 0~1
        img_gated = img_feat * gate                  (B, fd)
        fused     = cat([img_gated, proj_feat])      (B, 2*fd)
        logits    = classifier(fused)                (B, 1)   (无 sigmoid)
    """

    def __init__(
        self,
        pretrained_2d: bool = config_pretrained_2d,
        feature_dim: int = config_feature_dim,
        dropout_rate: float = config_dropout_rate,
        gate_hidden: int = 64,
    ):
        super(GatedDualStreamNet, self).__init__()
        self.image_stream = ImageStream(pretrained=pretrained_2d, feature_dim=feature_dim)
        self.proj_stream = ProjStream(feature_dim=feature_dim)
        self.gate_mlp = GateMLP(feature_dim=feature_dim, hidden=gate_hidden)

        # 拼接后分类头 (与 DualStreamStripNet 相同的 2*fd -> fd -> 1 结构)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 1)
            # 末尾不加 Sigmoid, 保持 logits 供 BCEWithLogitsLoss
        )

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor) -> torch.Tensor:
        img_feats = self.image_stream(img_tensor)          # (B, fd)
        proj_feats = self.proj_stream(proj_tensor)         # (B, fd)
        gate = self.gate_mlp(proj_feats)                   # (B, fd)
        img_gated = img_feats * gate                       # 门控调制
        fused = torch.cat([img_gated, proj_feats], dim=1)  # (B, 2*fd)
        return self.classifier(fused)

    def freeze_image_stream(self) -> None:
        """阶段一: 冻结 2D 图像流 backbone (仅训练投影流+门控+分类头+辅助头)。"""
        self.image_stream.freeze_backbone()

    def unfreeze_image_stream(self) -> None:
        """阶段二: 解冻 2D 图像流 backbone 微调。"""
        self.image_stream.unfreeze_backbone()


class GatedDualStreamAuxNet(nn.Module):
    """GatedDualStreamNet + 图像流条带位置辅助头的组合模型 (E 模式训练用)。

    复用 base 的 image_stream / proj_stream / gate_mlp / classifier, 一次前向同时
    给出主分类 logits (门控融合) 与辅助输出 (接在原始 img_feat 上, 不过门控)。
    权重保存仅存 base (不含辅助头), 与 D 分支的 best_model.pth 约定一致。
    """

    def __init__(self, base: GatedDualStreamNet, position_head: StripPositionHead):
        super(GatedDualStreamAuxNet, self).__init__()
        self.base = base
        self.position_head = position_head

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor):
        """返回 (logits, aux_logits)。"""
        img_feats = self.base.image_stream(img_tensor)          # (B, fd)
        proj_feats = self.base.proj_stream(proj_tensor)         # (B, fd)
        gate = self.base.gate_mlp(proj_feats)                   # (B, fd)
        img_gated = img_feats * gate
        logits = self.base.classifier(torch.cat([img_gated, proj_feats], dim=1))
        aux_logits = self.position_head(img_feats)              # 辅助接原始 img_feat (不门控)
        return logits, aux_logits


if __name__ == "__main__":
    # 模拟构建并前向测试
    base = GatedDualStreamNet(pretrained_2d=False)
    dummy_img = torch.randn(4, 3, 505, 220)
    dummy_proj = torch.randn(4, base.proj_stream.in_channels, 512)
    print("GatedDualStreamNet logits shape:", base(dummy_img, dummy_proj).shape)   # (4, 1)

    head = StripPositionHead(feature_dim=config_feature_dim)
    aux_net = GatedDualStreamAuxNet(base, head)
    logits, aux_logits = aux_net(dummy_img, dummy_proj)
    print("GatedDualStreamAuxNet logits shape:", logits.shape)     # (4, 1)
    print("GatedDualStreamAuxNet aux shape:   ", aux_logits.shape) # (4, 2)
