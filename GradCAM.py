"""
模型可解释性分析脚本（Grad-CAM）
功能：
  1. 加载训练好的 DualStreamStripNet 权重
  2. 对测试集（或指定图像）计算特征图梯度与权重
  3. 生成并在 outputs/heatmaps/ 中保存「原图 | 注意力热力图 | 叠加对比图」
"""
import os
import sys
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_val_transforms, get_projection_transforms
from src.models.dual_stream_net import DualStreamStripNet

# ================= 配置区 =================
TEST_CSV = "data\\test\\labels\\labels.csv"                     # 测试集标签 CSV
TEST_IMAGE_DIR = "data\\test\\raw"                             # 测试图片根目录
MODEL_WEIGHT_PATH = "outputs\\checkpoints\\P24\\best_model.pth"  # 选用哪一折的模型权重
CONFIG_PATH = "configs\\main_config.yaml"                     # 配置文件路径
OUTPUT_DIR = "outputs\\heatmaps"                              # 热力图输出目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 1                                             # 最多可视化多少张图（None 表示全部）
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载配置
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg['model']
    data_cfg = cfg['data']

    # 2. 实例化并加载模型权重
    print(f"正在加载模型: {MODEL_WEIGHT_PATH}")
    model = DualStreamStripNet(
        pretrained_2d=model_cfg.get('pretrained', False),
        feature_dim=model_cfg['feature_dim'],
        dropout_rate=model_cfg['dropout_rate']
    )
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 3. 自动定位 2D 骨干最后一层卷积（也可手动指定，例如 model.backbone_2d.layer4[-1]）
    target_layer = find_last_conv_layer(model)
    grad_cam = GradCAM(model, target_layer)

    # 4. 加载测试数据
    df = pd.read_csv(TEST_CSV)
    df['full_path'] = df['filename'].apply(lambda f: str(Path(TEST_IMAGE_DIR) / f))
    
    test_dataset = TestStripDataset(
        image_paths=df['full_path'].tolist(),
        labels=df['label'].tolist(),
        transforms=get_val_transforms(),
        projection_transforms=get_projection_transforms(),
        standard_dpi_size=tuple(data_cfg['standard_dpi_size']),
        proj_length=data_cfg['proj_length']
    )

    total_samples = len(test_dataset) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(test_dataset))
    print(f"开始生成热力图，共处理 {total_samples} 个样本...")

    # 5. 循环处理样本并可视化
    for idx in range(total_samples):
        img_tensor, proj_tensor, label = test_dataset[idx]
        
        # 增加 batch 维度
        img_input = img_tensor.unsqueeze(0).to(DEVICE)
        proj_input = proj_tensor.unsqueeze(0).to(DEVICE)

        # 生成 CAM 矩阵与预测概率
        cam_map, pred_prob = grad_cam.generate(img_input, proj_input)

        # 反归一化原图（用于画图显示）
        # [C, H, W] -> [H, W, C]，若有 Normalize 标准化可自行还原，此处做简单拉伸显示
        raw_img = img_tensor.permute(1, 2, 0).cpu().numpy()
        raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
        raw_img = np.uint8(255 * raw_img)

        # 生成伪彩色热力图 (JET 颜色映射)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 图像叠加: 原图 + 热力图 (alpha=0.6, beta=0.4)
        overlay = np.uint8(0.6 * raw_img + 0.4 * heatmap)

        # 6. 绘制 3 联对比图 (原图 / 热力图 / 叠加对比)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(raw_img)
        axes[0].set_title(f"Original Image\n(True Label: {label})", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(heatmap)
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (Attention Area)\nPred Prob: {pred_prob:.4f}", fontsize=12)
        axes[2].axis('off')

        # 7. 保存图像
        base_name = Path(df.iloc[idx]['filename']).stem
        pred_tag = "POS" if pred_prob >= 0.5 else "NEG"
        save_filename = f"cam_{idx:03d}_{base_name}_true_{label}_pred_{pred_tag}_{pred_prob:.2f}.png"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[{idx+1}/{total_samples}] 已保存: {save_filename}")

    print(f"\n所有热力图已生成完毕，存放在: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()