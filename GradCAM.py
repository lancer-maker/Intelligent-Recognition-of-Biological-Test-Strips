"""
单图 Grad-CAM 可解释性脚本
功能：
  1. 命令行输入单张图片 --image 与单个模型权重 --model-path
  2. 结构对齐 new_env_test.py：
       - 配置: src.utils.Fine_Tuning_helper.load_config
       - 权重: src.utils.model_helper.load_model_state
       - 模型: DualStreamStripNet(pretrained_2d=False, ...) (同 new_env_test.build_model)
       - 预处理: 复用 TestStripDataset 单样本管线 (行平均+中央ROI列平均双通道投影 / 2D DPI)
       - 确定性: 关闭投影随机增强 (与 new_env_test 默认一致, 结果可复现)
  3. 生成并保存「原图 | 注意力热力图 | 叠加对比图」到 outputs/heatmaps/ (保存位置不变)
  4. 判定阈值: 可用 --threshold 指定; 缺省自动读取权重同目录 metrics.csv 的 best_threshold
     (与 new_env_test.py 阈值来源一致), 找不到则回退 0.5

用法示例:
python GradCAM.py --image "data/test/raw/N_048_1.png"  --model-path "outputs/checkpoints/E/2/best_model.pth"
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.Fine_Tuning_helper import load_config       # 同 new_env_test.py
from src.utils.model_helper import load_model_state         # 同 new_env_test.py

# ================= 配置区 (保存位置不变) =================
TEST_LABELS_CSV = "data\\test\\labels\\labels.csv"   # 仅用于按文件名自动匹配真实标签(可选)
OUTPUT_DIR = "outputs\\heatmaps"                     # 热力图输出目录
# ==========================================


class GradCAM:
    """Grad-CAM 实现核心类"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册前向和后向 Hook
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor, proj_tensor):
        self.model.eval()
        self.model.zero_grad()

        # 1. 前向传播
        logits = self.model(img_tensor, proj_tensor)
        score = logits.squeeze()

        # 2. 反向传播计算梯度
        score.backward(retain_graph=True)

        # 3. 计算通道重要性权重 (Global Average Pooling on Gradients)
        gradients = self.gradients.detach()        # [1, C, H, W]
        activations = self.activations.detach()    # [1, C, H, W]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # 4. 加权求和并经过 ReLU
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        # 5. 上采样到输入图像分辨率
        cam = torch.nn.functional.interpolate(
            cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
            mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # 6. 归一化到 [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        prob = torch.sigmoid(logits).item()
        return cam, prob


def find_last_conv_layer(model):
    """自动寻找模型 2D 主干中的最后一个 Conv2d 层"""
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("未能自动在模型中找到 Conv2d 层，请手动指定 target_layer！")
    return last_conv


def parse_args():
    """命令行参数: 单张图片 + 单个模型权重 (风格对齐 new_env_test.py)。"""
    parser = argparse.ArgumentParser(
        description="单图 Grad-CAM: 输入单张图片 + 单个模型权重, 结构对齐 new_env_test.py"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="单张测试图片路径 (绝对或相对均可)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="单个模型权重路径 (裸 state_dict 或含 model_state_dict 的 ckpt 均可, 经 load_model_state)")
    parser.add_argument("--label", type=int, default=None,
                        help="真实标签 0/1 (可选; 缺省则按文件名从测试集 CSV 自动匹配, 匹配不到显示 NA)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="判定阈值 (缺省自动读取模型同目录 metrics.csv 的 best_threshold,\n"
                             "与 new_env_test.py 相同; 找不到则回退 0.5)")
    return parser.parse_args()


def resolve_label_from_csv(image_path: str):
    """按 basename 在测试集 labels.csv 中查找真实标签, 找不到返回 None。"""
    try:
        df = pd.read_csv(TEST_LABELS_CSV)
    except Exception:
        return None
    hit = df.loc[df["filename"] == Path(image_path).name, "label"]
    return int(hit.iloc[0]) if len(hit) else None


def resolve_threshold(weight_path: str, user_threshold: Optional[float]) -> float:
    """判定阈值: 优先 --threshold; 否则读权重同目录 metrics.csv 的 best_threshold (同 new_env_test); 回退 0.5。"""
    if user_threshold is not None:
        return float(user_threshold)
    try:
        metrics_csv = Path(weight_path).parent / "metrics.csv"
        if metrics_csv.exists():
            row = pd.read_csv(metrics_csv).iloc[0]
            th = row.get("best_threshold")
            if th is not None and pd.notna(th):
                return float(th)
    except Exception:
        pass
    return 0.5


def build_single_sample(image_path: str, label, data_cfg: dict):
    """构造单张样本 (img_tensor, proj_tensor)。确定性: 关闭投影随机增强。

    TestStripDataset 内部依据配置 proj_use_col_avg 自动决定投影通道数
    (行平均 + 中央ROI列平均双通道); projection_transforms=False 表示不施加
    投影亮度/对比度随机增强 -> 输出确定可复现 (与 new_env_test 默认一致)。
    """
    ds = TestStripDataset(
        image_paths=[str(image_path)],
        labels=[0 if label is None else int(label)],
        transforms=get_val_transforms(),
        projection_transforms=False,
        standard_dpi_size=tuple(data_cfg["standard_dpi_size"]),
        proj_length=int(data_cfg["proj_length"]),
    )
    img_tensor, proj_tensor, _ = ds[0]
    return img_tensor, proj_tensor


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 配置: 与 new_env_test 相同 (load_config 读取 main_config.yaml)
    cfg = load_config()
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 构建模型 + 加载权重 (同 new_env_test.build_model / load_model_state)
    print(f"正在加载模型: {args.model_path}")
    model = DualStreamStripNet(
        pretrained_2d=False,
        feature_dim=int(model_cfg["feature_dim"]),
        dropout_rate=float(model_cfg["dropout_rate"]),
    ).to(device)
    model = load_model_state(model, str(args.model_path))
    model.eval()

    # 3. 自动定位 2D 骨干最后一层卷积 (Grad-CAM 目标层)
    target_layer = find_last_conv_layer(model)
    grad_cam = GradCAM(model, target_layer)

    # 4. 真实标签 + 判定阈值
    label = args.label if args.label is not None else resolve_label_from_csv(args.image)
    label_text = str(label) if label is not None else "NA"
    threshold = resolve_threshold(args.model_path, args.threshold)
    print(f"图片: {args.image} | 真实标签: {label_text} | 设备: {device}")
    print(f"判定阈值: {threshold:.4f} (来源: {'--threshold' if args.threshold is not None else 'metrics.csv best_threshold(同new_env_test)'} 或回退0.5)")

    # 5. 单样本预处理 (确定性: 关闭投影随机增强)
    img_tensor, proj_tensor = build_single_sample(args.image, label, data_cfg)
    img_input = img_tensor.unsqueeze(0).to(device)
    proj_input = proj_tensor.unsqueeze(0).to(device)

    # 6. 生成 CAM 矩阵与预测概率
    cam_map, pred_prob = grad_cam.generate(img_input, proj_input)

    # 7. 反归一化原图 (用于显示)
    # [C, H, W] -> [H, W, C]，此处做简单拉伸显示
    raw_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
    raw_img = np.uint8(255 * raw_img)

    # 8. 伪彩色热力图 (JET) + 叠加
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * raw_img + 0.4 * heatmap)

    # 9. 绘制 3 联对比图 (原图 / 热力图 / 叠加对比)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Original Image\n(True Label: {label_text})", fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (Attention Area)\nPred Prob: {pred_prob:.4f} (thr={threshold:.4f})", fontsize=12)
    axes[2].axis('off')

    # 10. 保存图像 (保存位置不变: OUTPUT_DIR = outputs/heatmaps; 判定使用与 new_env_test 一致的阈值)
    base_name = Path(args.image).stem
    pred_tag = "POS" if pred_prob >= threshold else "NEG"
    save_filename = f"cam_000_{base_name}_true_{label_text}_pred_{pred_tag}_{pred_prob:.2f}.png"
    save_path = os.path.join(OUTPUT_DIR, save_filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


if __name__ == "__main__":
    main()