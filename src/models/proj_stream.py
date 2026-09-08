# 1D 物理投影信号流，基于自定义 4 层 1D-CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))        #将项目根目录加入搜索范围
from configs.load_config import get_main_config

config = get_main_config()
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
# --- 投影流输入通道推断 (仅信号通道; 位置编码由 ProjStream 内部实现) ---
# 基础: 行平均(纵向)投影 = 1 个通道
# 可选: +1 中央ROI列平均(横向)投影通道 (data.proj_use_col_avg)
config_proj_use_col_avg = bool(config.get("data", {}).get("proj_use_col_avg", False))
config_proj_signal_channels = 1 + int(config_proj_use_col_avg)   # 数据集提供的信号通道数(不含位置)
config_proj_length = int(config.get("data", {}).get("proj_length", 512))
# --- 位置编码总开关 (model.proj_add_position_channel) ---
# true : 开启"可学习位置编码" —— CNN 输入上拼接一个可学习位置通道 nn.Parameter(1,1,L),
#        完全由训练学习 (不保留任何固定坐标输入/输出分支), 首卷积输入通道 = 信号 + 1;
# false: 关闭位置编码, 仅使用信号通道。
config_proj_add_position = bool(config.get("model", {}).get("proj_add_position_channel", False))
config_proj_dropout = float(config.get("model", {}).get("proj_dropout_rate", 0.6))
# --- 投影流 1D CNN 结构 (核大小 / 输出通道, 默认 4 层, 由配置给出) ---
config_proj_kernel_sizes = [int(k) for k in config.get("model", {}).get("proj_conv_kernel_sizes", [15, 9, 5, 3])]
config_proj_conv_channels = [int(c) for c in config.get("model", {}).get("proj_conv_channels", [32, 64, 128, 256])]


class ProjStream(nn.Module):
    """
    1D 投影流特征提取网络。
    通过加深加宽的 1D-CNN (默认 4 层: 核 15/9/5/3, 通道 32->64->128->256)
    捕捉试纸条带物理密度峰值与形态信息; 更大的起始卷积核扩大感受野,
    可覆盖更宽的条带形态特征。

    位置编码 (model.proj_add_position_channel 总开关):
      - 外部输入仅提供"信号通道": 行平均(1) + 列平均可选(1);
      - true : CNN 输入上额外拼接一个"可学习位置通道" nn.Parameter(1,1,L),
               零初始化、完全随训练学习; 不保留任何固定坐标输入/输出分支;
      - false: 不启用位置编码。

    外部需提供的输入通道数 (in_channels / signal_channels):
      - 1: 仅行平均(纵向)投影 (1, 512)
      - 2: 行平均 + 中央ROI列平均(横向) 双通道 (2, 512)
    """

    def __init__(self, feature_dim: int = config_feature_dim, in_channels: int = None):
        super(ProjStream, self).__init__()
        if in_channels is None:
            in_channels = config_proj_signal_channels
        self.in_channels = int(in_channels)   # 外部输入信号通道数 (不含位置通道)
        self.use_position = config_proj_add_position

        kernel_sizes = list(config_proj_kernel_sizes)
        conv_channels = list(config_proj_conv_channels)
        if len(kernel_sizes) != len(conv_channels):
            raise ValueError(
                f"proj_conv_kernel_sizes 与 proj_conv_channels 长度不一致: "
                f"{len(kernel_sizes)} vs {len(conv_channels)}"
            )

        # 第一个卷积块的实际输入通道 = 信号通道 + (启用时) 1 个可学习位置通道
        conv_in = self.in_channels + (1 if self.use_position else 0)
        # 输入形状: (B, conv_in, 512) -> 512 个点均匀映射物理长度 5.05 mm
        # 每个卷积块 = Conv1d + BN + ReLU; 除最后一层外每块后接 MaxPool(k=2)
        blocks: list = []
        cin = conv_in
        for i, (k, cout) in enumerate(zip(kernel_sizes, conv_channels)):
            blocks.append(nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2))
            blocks.append(nn.BatchNorm1d(cout))
            blocks.append(nn.ReLU())
            if i < len(kernel_sizes) - 1:              # 最后一个卷积块后不接 MaxPool
                blocks.append(nn.MaxPool1d(kernel_size=2))
            cin = cout

        # 全局平均池化 -> (B, conv_channels[-1], 1), 压缩空间维度
        blocks.append(nn.AdaptiveAvgPool1d(1))
        self.conv_layers = nn.Sequential(*blocks)

        # ---- 可学习位置编码 (无固定坐标, 完全由训练学习) ----
        if self.use_position:
            # 可学习位置通道: 长度 = 投影流长度 L; 零初始化后随训练学习
            self.learned_pos = nn.Parameter(torch.zeros(1, 1, config_proj_length))

        # 全局平均池化后的特征维度即 fc 输入维度
        fc_in = conv_channels[-1]

        # Dropout(0.6) 防过拟合 + 降维全连接层 -> (B, feature_dim)
        self.dropout = nn.Dropout(p=config_proj_dropout)
        self.fc = nn.Linear(fc_in, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为 (B, C, L) 的信号投影 Tensor (C = 1 / 2, 不含位置通道)
        Returns:
            out: 形状为 (B, feature_dim) 的投影特征向量
        """
        # 1) 在输入信号上拼接可学习位置通道
        if self.use_position:
            pos = self.learned_pos.expand(x.shape[0], 1, -1)   # (B, 1, L)
            x = torch.cat([x, pos], dim=1)

        feats = self.conv_layers(x)                # (B, conv_channels[-1], 1)
        feats = torch.flatten(feats, 1)            # (B, conv_channels[-1])

        out = F.relu(self.fc(self.dropout(feats)))  # (B, feature_dim)
        return out