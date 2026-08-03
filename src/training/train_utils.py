# 提供损失函数构建、优化器配置以及梯度更新相关的实用工具函数
import torch
import torch.nn as nn
import torch.optim as optim


def get_weighted_bce_loss(pos_weight_value: float, device: torch.device) -> nn.Module:
    """
    构建带阳性样本权重的 BCEWithLogitsLoss。
    
    Args:
        pos_weight_value: 阳性样本权重 = (阴性样本数 / 阳性样本数)
        device: 计算设备 (CPU 或 CUDA)
        
    Returns:
        nn.Module: 配置好的损失函数实例
    """
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_optimizer(
    model: nn.Module, 
    lr: float = 1e-4, 
    weight_decay: float = 1e-3
) -> optim.Optimizer:
    """
    为模型中当前可求导的参数 (requires_grad=True) 构建 AdamW 优化器。
    
    Args:
        model: 模型实例
        lr: 初始学习率
        weight_decay: 权重衰减系数
        
    Returns:
        optim.Optimizer: 配置好的 AdamW 优化器
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)