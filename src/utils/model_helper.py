import torch
from typing import Union


def load_model_state(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """从 checkpoint 路径加载模型权重，并返回模型。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def load_model_state_dict(checkpoint_path: str) -> dict:
    """读取模型权重字典，适配保存为 state_dict 或 model_state_dict 的情况。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint if isinstance(checkpoint, dict) else {}
