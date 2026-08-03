# 1D 物理投影信号流，基于自定义 3 层 1D-CNN
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

class ProjStream(nn.Module):
    """
    1D 投影流特征提取网络。
    通过 3 层 1D 卷积捕捉试纸条带物理密度峰值信号。
    """
    def __init__(self, feature_dim: int = config_feature_dim):
        super(ProjStream, self).__init__()
        
        # 输入形状: (B, 1, 512) -> 512 个点均匀映射物理长度 5.05 mm
        self.conv_layers = nn.Sequential(
            # Stage 1: (B, 1, 512) -> (B, 32, 256)
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Stage 2: (B, 32, 256) -> (B, 64, 128)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Stage 3: (B, 64, 128) -> (B, 128, 64)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # 全局平均池化 -> (B, 128, 1)
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 降维全连接层 -> (B, 128)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为 (B, 1, 512) 的一维物理投影 Tensor
        Returns:
            out: 形状为 (B, 128) 的投影特征向量
        """
        feats = self.conv_layers(x)         # (B, 128, 1)
        feats = torch.flatten(feats, 1)     # (B, 128)
        out = F.relu(self.fc(feats))        # (B, 128)
        return out