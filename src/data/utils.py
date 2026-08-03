import cv2
import numpy as np
import torch
from typing import Tuple, Optional


def read_image_rgb(image_path: str, target_size: Optional[Tuple[int, int]] = (220, 505)) -> np.ndarray:
    """
    读取图像并统一到标准物理 DPI 分辨率 (220, 505)。
    对于低分辨率图像，使用 INTER_CUBIC 高阶插值减少伪影。
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    if target_size is not None:
        h, w = img_rgb.shape[:2]
        target_w, target_h = target_size
        
        # 如果尺寸不一致，选择物理感知更好的双三次插值 (INTER_CUBIC)
        if (w, h) != (target_w, target_h):
            # 低分辨率放大用 INTER_CUBIC，高分辨率缩小用 INTER_AREA
            interpolation = cv2.INTER_CUBIC if w < target_w else cv2.INTER_AREA
            img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=interpolation)
            
    return img_rgb


def extract_1d_projection_resampled(image_rgb_raw: np.ndarray, target_length: int = 512) -> torch.Tensor:
    """
    【核心修正】从原始任意分辨率图像中提取 1D 投影，并线性重采样到固定物理采样点数 (如 512)。
    
    物理意义: 512 个点均匀映射物理长度 5.05 mm，每点代表 ~0.0098 mm。
    
    Args:
        image_rgb_raw: 原始未缩放的 RGB 图像 (H_raw, W_raw, 3)
        target_length: 重采样后的固定 1D 序列长度 (默认 512)
        
    Returns:
        torch.Tensor: 形状为 (1, target_length) 的物理归一化 1D 投影 Tensor
    """
    # 1. 转灰度图
    gray = cv2.cvtColor(image_rgb_raw, cv2.COLOR_RGB2GRAY)
    
    # 2. 沿原始宽度求均值，得到原始像素长度的 1D 投影 (长度为 H_raw)
    raw_proj = np.mean(gray, axis=1, dtype=np.float32)
    raw_length = len(raw_proj)
    
    # 3. Min-Max 基础归一化
    raw_proj_norm = (raw_proj - raw_proj.min()) / (raw_proj.max() - raw_proj.min() + 1e-6)
    
    # 4. 【关键步骤】一维物理空间线性插值重采样 (Linear Interpolation)
    if raw_length != target_length:
        x_raw = np.linspace(0, 5.05, num=raw_length)       # 原始物理坐标 (mm)
        x_target = np.linspace(0, 5.05, num=target_length) # 目标物理坐标 (mm)
        # 线性插值重采样
        resampled_proj = np.interp(x_target, x_raw, raw_proj_norm)
    else:
        resampled_proj = raw_proj_norm

    # 5. 转为 Tensor 并增加 Channel 维 -> (1, target_length)
    proj_tensor = torch.from_numpy(resampled_proj).float().unsqueeze(0)
    
    return proj_tensor