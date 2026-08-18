"""
加载模型
读取分类头第一层权重：classifier[0].weight
将 256 维输入特征按前 128 维和后 128 维拆成两部分
分别计算 L1 与 L2 范数，比较二者的权重大小
根据结果给出“图像流更重要 / 投影流更重要 / 两者相近”的提示
"""
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.dual_stream_net import DualStreamStripNet

# 配置
MODEL_WEIGHTS = "outputs/checkpoints/P24/best_model.pth"

def main():
    # 加载模型
    model = DualStreamStripNet(pretrained_2d=False, feature_dim=128, dropout_rate=0.5)
    checkpoint = torch.load(MODEL_WEIGHTS, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)

    # 提取第一层全连接权重
    fc_weight = model.classifier[0].weight.data  # shape: (128, 256)
    print("权重矩阵形状:", fc_weight.shape)

    # 前128列对应图像流特征，后128列对应投影流特征
    img_part = fc_weight[:, :128]
    proj_part = fc_weight[:, 128:]

    img_l1 = img_part.abs().sum().item()
    proj_l1 = proj_part.abs().sum().item()
    img_l2 = torch.norm(img_part).item()
    proj_l2 = torch.norm(proj_part).item()

    print("=" * 60)
    print("分类头第一层权重统计：")
    print(f"图像流部分 L1 范数: {img_l1:.4f}")
    print(f"投影流部分 L1 范数: {proj_l1:.4f}")
    print(f"图像流部分 L2 范数: {img_l2:.4f}")
    print(f"投影流部分 L2 范数: {proj_l2:.4f}")
    print("=" * 60)

    if proj_l1 > img_l1 * 3:
        print("提示：模型明显更依赖投影流，图像流可能被忽略。")
    elif img_l1 > proj_l1 * 3:
        print("提示：模型明显更依赖图像流，投影流可能被忽略。")
    else:
        print("两个流的权重分布相对均衡。")

if __name__ == "__main__":
    main()