"""
提取单张试纸图像的双流特征并打印
用法：修改下方配置区的图像路径、权重路径等参数后直接运行
"""
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import cv2

# 将项目根目录加入路径，确保能导入 src 模块
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.utils import (
    read_image_rgb,
    extract_1d_projection_resampled,
    extract_col_avg_projection_resampled,
)
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

# ================= 配置区 =================
IMAGE_PATH = r"data\test\raw\N_023_0.jpg"           # 待提取特征的图像路径
# 注意: 必须是模型权重 .pth 文件, 不能是 .npy 概率文件
MODEL_WEIGHTS = r"outputs\checkpoints\Tweak\3\best_model.pth"  # 模型权重路径
CONFIG_PATH = "configs/main_config.yaml"           # 主配置文件
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 1. 读取配置
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg['model']
    data_cfg = cfg['data']

    # 2. 加载模型
    model = DualStreamStripNet(
        pretrained_2d=model_cfg['pretrained'],
        feature_dim=model_cfg['feature_dim'],
        dropout_rate=model_cfg['dropout_rate']
    )
    load_model_state(model, MODEL_WEIGHTS)
    model.to(DEVICE)
    model.eval()

    # 3. 读取并预处理图像
    # 3.1 读取原始 RGB 图像（不缩放）
    raw_rgb = read_image_rgb(IMAGE_PATH)   # 返回 RGB 格式的原始图像 (H, W, 3)

    # 3.2 提取投影（基于原始图像，与训练一致）; 适配双通道(行平均+可选中央ROI列平均)配置
    use_col_avg = bool(data_cfg.get("proj_use_col_avg", False))
    proj_tensor = extract_1d_projection_resampled(
        raw_rgb, target_length=data_cfg['proj_length']
    )  # 形状 (1, proj_length)
    if use_col_avg:
        col_tensor = extract_col_avg_projection_resampled(
            raw_rgb,
            target_length=int(data_cfg.get("proj_col_length", 512)),
            roi_fraction=float(data_cfg.get("proj_col_roi_fraction", 0.33)),
        )
        proj_tensor = torch.cat([proj_tensor, col_tensor], dim=0)  # (2, proj_length)

    # 3.3 图像流预处理：缩放到标准尺寸，进行验证集变换（归一化+转Tensor）
    target_w, target_h = data_cfg['standard_dpi_size']
    h, w = raw_rgb.shape[:2]
    if (w, h) != (target_w, target_h):
        interp = cv2.INTER_CUBIC if w < target_w else cv2.INTER_AREA
        dpi_aligned_rgb = cv2.resize(raw_rgb, (target_w, target_h), interpolation=interp)
    else:
        dpi_aligned_rgb = raw_rgb

    val_transforms = get_val_transforms()
    augmented = val_transforms(image=dpi_aligned_rgb)
    image_tensor = augmented['image']  # (3, 505, 220)

    # 4. 添加 batch 维度并移动到设备
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    proj_tensor = proj_tensor.unsqueeze(0).to(DEVICE)

    # 5. 分别提取两个流的特征
    with torch.no_grad():
        # 图像流：ImageStream 已经直接输出 128 维特征
        img_features = model.image_stream(image_tensor)          # (B, 128)

        # 投影流：ProjStream 已经直接输出 128 维特征
        proj_features = model.proj_stream(proj_tensor)           # (B, 128)

    # 6. 转换为 CPU 并输出数值
    img_vec = img_features.squeeze(0).cpu().numpy()
    proj_vec = proj_features.squeeze(0).cpu().numpy()

    torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)
    print("=" * 80)
    print("图像流特征 (ImageStream 128-d):")
    print(torch.from_numpy(img_vec))
    print("\n投影流特征 (ProjStream 128-d):")
    print(torch.from_numpy(proj_vec))
    print("=" * 80)

if __name__ == "__main__":
    main()