import torch
from typing import Union


def load_model_state(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """从 checkpoint 路径加载模型权重，并返回模型。"""
    # PyTorch 2.6+ 默认 weights_only=True, 对本地训练保存的权重 (可能含旧版/自定义类型)
    # 直接加载会抛 UnpicklingError; 本仓库权重均为本地训练产出, 属可信来源, 故显式设为 False。
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def load_model_state_dict(checkpoint_path: str) -> dict:
    """读取模型权重字典，适配保存为 state_dict 或 model_state_dict 的情况。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint if isinstance(checkpoint, dict) else {}
