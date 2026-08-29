"""
双流特征消融实验：分别禁用图像流 / 投影流，观察预测变化
"""
import os
import sys
import yaml
import torch
import cv2
from pathlib import Path

# 加入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.utils import read_image_rgb, extract_1d_projection_resampled
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

# ================= 配置区 =================
IMAGE_PATH = "data/test/raw/N_026_1.jpg"        # 待分析图像（如漏检阳性样本）
MODEL_WEIGHTS = "outputs/checkpoints/P24/best_model.pth"  # 某个模型权重
CONFIG_PATH = "configs/main_config.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg['model']
    data_cfg = cfg['data']

    # 加载模型
    model = DualStreamStripNet(
        pretrained_2d=model_cfg['pretrained'],
        feature_dim=model_cfg['feature_dim'],
        dropout_rate=model_cfg['dropout_rate']
    )
    load_model_state(model, MODEL_WEIGHTS)
    model.to(DEVICE)
    model.eval()

    # 读取原始图像
    raw_rgb = read_image_rgb(IMAGE_PATH)

    # 提取投影
    proj_tensor = extract_1d_projection_resampled(raw_rgb, target_length=data_cfg['proj_length'])
    proj_tensor = proj_tensor.unsqueeze(0).to(DEVICE)  # (1, 1, proj_length)

    # 图像流预处理
    target_w, target_h = data_cfg['standard_dpi_size']
    h, w = raw_rgb.shape[:2]
    if (w, h) != (target_w, target_h):
        interp = cv2.INTER_CUBIC if w < target_w else cv2.INTER_AREA
        dpi_aligned_rgb = cv2.resize(raw_rgb, (target_w, target_h), interpolation=interp)
    else:
        dpi_aligned_rgb = raw_rgb
    val_transforms = get_val_transforms()
    augmented = val_transforms(image=dpi_aligned_rgb)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)  # (1, 3, 505, 220)

    # 提取两个流的特征
    with torch.no_grad():
        img_feats = model.image_stream(image_tensor)  # (1, 128)
        proj_feats = model.proj_stream(proj_tensor)  # (1, 128)

        # 正常预测
        fused_normal = torch.cat((img_feats, proj_feats), dim=1)
        logits_normal = model.classifier(fused_normal)
        prob_normal = torch.sigmoid(logits_normal).item()

        # 仅投影流（图像特征置零）
        img_zero = torch.zeros_like(img_feats)
        fused_only_proj = torch.cat((img_zero, proj_feats), dim=1)
        logits_only_proj = model.classifier(fused_only_proj)
        prob_only_proj = torch.sigmoid(logits_only_proj).item()

        # 仅图像流（投影特征置零）
        proj_zero = torch.zeros_like(proj_feats)
        fused_only_img = torch.cat((img_feats, proj_zero), dim=1)
        logits_only_img = model.classifier(fused_only_img)
        prob_only_img = torch.sigmoid(logits_only_img).item()

    print("=" * 60)
    print(f"图像: {IMAGE_PATH}")
    print(f"正常预测概率: {prob_normal:.4f}")
    print(f"仅投影流概率: {prob_only_proj:.4f}")
    print(f"仅图像流概率: {prob_only_img:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()