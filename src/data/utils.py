import cv2
import numpy as np
import torch
from typing import Tuple, Optional

try:
    from scipy.signal import detrend
except ImportError:  # pragma: no cover
    detrend = None


def resize_to_standard_dpi(image_rgb: np.ndarray, target_size: Tuple[int, int] = (220, 505)) -> np.ndarray:
    """将 RGB 图像统一到目标 DPI 尺寸，保留物理感知的插值策略。"""
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"期望输入为 (H, W, 3) 的 RGB 图像，实际形状为 {image_rgb.shape}")

    h, w = image_rgb.shape[:2]
    target_w, target_h = target_size
    if (w, h) != (target_w, target_h):
        interpolation = cv2.INTER_CUBIC if w < target_w else cv2.INTER_AREA
        return cv2.resize(image_rgb, (target_w, target_h), interpolation=interpolation)
    return image_rgb


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
        img_rgb = resize_to_standard_dpi(img_rgb, target_size=target_size)

    return img_rgb


def normalize_projection_baseline(signal: np.ndarray) -> np.ndarray:
    """去基线：中心化并在存在漂移时去掉线性趋势，突出微弱波动。"""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal

    # 先去均值，避免整体幅值主导卷积响应
    centered = signal - np.mean(signal)

    # 若可用 scipy，当信号存在线性漂移时进一步去趋势
    if detrend is not None:
        try:
            centered = detrend(centered, type='linear', overwrite=False)
        except TypeError:
            centered = detrend(centered, type='linear')

    return centered


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

    # 5. 去基线：让 1D CNN 关注波动，而不是整体幅值大小
    resampled_proj = normalize_projection_baseline(resampled_proj)

    # 6. 转为 Tensor 并增加 Channel 维 -> (1, target_length)
    proj_tensor = torch.from_numpy(resampled_proj).float().unsqueeze(0)
    
    return proj_tensor


def extract_col_avg_projection_resampled(
    image_rgb_raw: np.ndarray,
    target_length: int = 512,
    roi_fraction: float = 0.33,
) -> torch.Tensor:
    """
    【列平均(横向)投影】仅取中央 ROI 区域 (高度方向中间 roi_fraction), 对每一列沿高度求平均。

    与行平均投影 (extract_1d_projection_resampled) 互补:
      - 行平均(纵向): 捕捉 T/C 线在纵向 (高度/行) 的位置信息, 物理长度对应图像高度 5.05mm
      - 列平均(横向): 仅中央 ROI 内沿高度求平均, 反映条带/色块沿横向 (宽度/列) 的分布,
        物理长度对应试纸条宽度 2.2mm; 两者物理尺度不同, 作为特征输入网络可自行学习

    Args:
        image_rgb_raw: 原始 RGB 图像 (H, W, 3)
        target_length: 重采样后的固定长度 (建议与行平均一致, 如 512)
        roi_fraction: 中央 ROI 占高度方向的比例 (如 0.33 表示中间 1/3)

    Returns:
        torch.Tensor: 形状 (1, target_length) 的横向投影 Tensor
    """
    gray = cv2.cvtColor(image_rgb_raw, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    # 1. 确定中央 ROI (高度方向中间 roi_fraction)
    roi_h = max(1, int(round(h * roi_fraction)))
    start_y = (h - roi_h) // 2
    roi = gray[start_y:start_y + roi_h, :]

    # 2. 对每一列沿高度(axis=0)求平均 -> 长度 = 宽度 W
    raw_proj = np.mean(roi, axis=0, dtype=np.float32)
    raw_length = len(raw_proj)

    # 3. Min-Max 基础归一化
    raw_proj_norm = (raw_proj - raw_proj.min()) / (raw_proj.max() - raw_proj.min() + 1e-6)

    # 4. 一维线性插值重采样到固定长度 (物理尺度按试纸宽度 2.2mm 注释)
    if raw_length != target_length:
        x_raw = np.linspace(0, 2.2, num=raw_length)
        x_target = np.linspace(0, 2.2, num=target_length)
        resampled_proj = np.interp(x_target, x_raw, raw_proj_norm)
    else:
        resampled_proj = raw_proj_norm

    # 5. 去基线 (与行平均保持相同的预处理)
    resampled_proj = normalize_projection_baseline(resampled_proj)

    # 6. 转 Tensor -> (1, target_length)
    return torch.from_numpy(resampled_proj).float().unsqueeze(0)