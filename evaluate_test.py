"""
独立测试集集成与逐折模型评估脚本
输出：
  1. test_individual_model_metrics.csv —— 格式为: patient_id,best_threshold,auc,accuracy,sensitivity,specificity
  2. test_ensemble_predictions.csv     —— 预测详情
  3. test_ensemble_summary.csv         —— 集成模型总指标
"""
import os
import sys
import glob
import re
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_val_transforms, get_projection_transforms
from src.models.dual_stream_net import DualStreamStripNet

# ================= 配置区 =================
TEST_CSV = "data\\test\\labels\\labels.csv"                    # 测试集标签文件
TEST_IMAGE_DIR = "data\\test\\raw"                            # 测试图像所在文件夹
CHECKPOINT_DIR_PATTERN = "outputs\\checkpoints\\P*\\best_model.pth"  # 权重通配符
CONFIG_PATH = "configs\\main_config.yaml"                    # 配置文件
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
OUTPUT_DIR = "outputs\\reports\\test"
# ==========================================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def compute_metrics_with_best_threshold(labels, probs):
    """
    通过约登指数 (Youden's Index) 自动寻找最佳分类阈值，并计算对应指标
    """
    if len(np.unique(labels)) < 2:
        auc = np.nan
        best_threshold = 0.5
    else:
        auc = roc_auc_score(labels, probs)
        fpr, tpr, thresholds = roc_curve(labels, probs)
        
        # 约登指数: J = sensitivity + specificity - 1 = tpr - fpr
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = float(thresholds[best_idx])
        
        # sklearn 中 thresholds[0] 可能会设置为 max(probs) + 1，做截断处理
        if best_threshold > 1.0:
            best_threshold = float(np.max(probs))

    # 使用最佳阈值得到二分类预测
    preds = (probs >= best_threshold).astype(int)
    
    # 混淆矩阵与性能指标
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    accuracy = accuracy_score(labels, preds)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'best_threshold': best_threshold,
        'auc': float(auc),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }

def extract_patient_id(fold_name):
    """从文件夹名称（如 P01, P1, Fold_1）中提取纯数字 ID"""
    match = re.search(r'\d+', fold_name)
    if match:
        return int(match.group())
    return fold_name

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 读取配置
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg['model']
    data_cfg = cfg['data']

    # 2. 加载测试数据
    df = pd.read_csv(TEST_CSV)
    df['filename'] = df['filename'].apply(lambda f: str(Path(TEST_IMAGE_DIR) / f))
    print(f"测试集样本数: {len(df)}")

    test_dataset = TestStripDataset(
        image_paths=df['filename'].tolist(),
        labels=df['label'].tolist(),
        transforms=get_val_transforms(),
        projection_transforms=get_projection_transforms(),
        standard_dpi_size=tuple(data_cfg['standard_dpi_size']),
        proj_length=data_cfg['proj_length']
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 查找所有模型权重
    weight_paths = sorted(glob.glob(CHECKPOINT_DIR_PATTERN))
    if not weight_paths:
        raise FileNotFoundError(f"未找到符合模式 {CHECKPOINT_DIR_PATTERN} 的权重文件")
    print(f"找到 {len(weight_paths)} 个模型权重文件")

    # 4. 逐个模型预测
    all_probs = []
    model_names = []
    individual_metrics = []
    labels = df['label'].values.astype(int)

    for path in weight_paths:
        fold_name = Path(path).parent.name
        patient_id = extract_patient_id(fold_name)
        model_names.append(fold_name)
        print(f"\n>>> 正在评估 Patient/Fold: {patient_id} ({fold_name})")

        # 实例化模型
        model = DualStreamStripNet(
            pretrained_2d=model_cfg['pretrained'],
            feature_dim=model_cfg['feature_dim'],
            dropout_rate=model_cfg['dropout_rate']
        )
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        # 预测概率
        probs = []
        with torch.no_grad():
            for img_batch, proj_batch, _ in test_loader:
                img_batch = img_batch.to(DEVICE)
                proj_batch = proj_batch.to(DEVICE)
                logits = model(img_batch, proj_batch)
                prob = torch.sigmoid(logits).cpu().numpy().flatten()
                probs.extend(prob)
        probs = np.array(probs)
        all_probs.append(probs)

        # 基于约登指数计算该折最佳阈值及指标
        metrics = compute_metrics_with_best_threshold(labels, probs)
        metrics['patient_id'] = patient_id
        individual_metrics.append(metrics)

        print(f"  Thresh={metrics['best_threshold']:.4f}, AUC={metrics['auc']:.4f}, "
              f"Acc={metrics['accuracy']:.4f}, Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")

    # 5. 保存你要求的格式文件 (test_individual_model_metrics.csv)
    individual_df = pd.DataFrame(individual_metrics)
    # 按照数字排序并重排列顺序
    if 'patient_id' in individual_df.columns and pd.api.types.is_numeric_dtype(individual_df['patient_id']):
        individual_df = individual_df.sort_values(by='patient_id')
    
    # 保持列顺序与需求完全一致
    columns_order = ['patient_id', 'best_threshold', 'auc', 'accuracy', 'sensitivity', 'specificity']
    individual_df = individual_df[columns_order]
    
    individual_csv_path = os.path.join(OUTPUT_DIR, "test_individual_model_metrics.csv")
    individual_df.to_csv(individual_csv_path, index=False)
    print(f"\n[已生成] 单折指标汇总已保存至: {individual_csv_path}")

    # 6. 集成概率与汇总输出 (可选保留)
    probs_matrix = np.stack(all_probs, axis=1)
    ensemble_probs = probs_matrix.mean(axis=1)
    
    ensemble_metrics = compute_metrics_with_best_threshold(labels, ensemble_probs)
    ensemble_metrics['num_models'] = len(weight_paths)
    
    summary_csv_path = os.path.join(OUTPUT_DIR, "test_ensemble_summary.csv")
    pd.DataFrame([ensemble_metrics]).to_csv(summary_csv_path, index=False)

    df_out = df.copy()
    for i, name in enumerate(model_names):
        df_out[f'prob_{name}'] = probs_matrix[:, i]
    df_out['ensemble_prob'] = ensemble_probs
    df_out['ensemble_pred_label'] = (ensemble_probs >= ensemble_metrics['best_threshold']).astype(int)
    
    pred_csv_path = os.path.join(OUTPUT_DIR, "test_ensemble_predictions.csv")
    df_out.to_csv(pred_csv_path, index=False)

if __name__ == "__main__":
    main()