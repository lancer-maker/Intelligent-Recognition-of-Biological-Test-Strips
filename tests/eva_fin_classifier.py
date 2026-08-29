"""
单模型评估脚本：仅评估微调后的分类头模型
支持双阈值对比输出 (最佳 Youden's J 阈值 vs 标准 0.5 阈值)
固定种子后的版本
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_projection_transforms, get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

TEST_CSV = PROJECT_ROOT / "data" / "test" / "labels" / "labels.csv"
TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "test" / "raw"
MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "Tweak" / "finetuned_classifier.pth"
CONFIG_PATH = PROJECT_ROOT / "configs" / "main_config.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "test"


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics_with_best_threshold(labels, probs):
    # ================= 核心修改：同时计算两个阈值的指标 =================
    if len(np.unique(labels)) < 2:
        auc = np.nan
        best_threshold = 0.5
    else:
        auc = roc_auc_score(labels, probs)
        fpr, tpr, thresholds = roc_curve(labels, probs)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(thresholds[best_idx])
        if best_threshold > 1.0:
            best_threshold = float(np.max(probs))

    # 计算基于 best_threshold 的指标
    preds_best = (probs >= best_threshold).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(labels, preds_best, labels=[0, 1]).ravel()
    acc_b = accuracy_score(labels, preds_best)
    sens_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
    spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0

    # 计算基于标准阈值 0.5 的指标
    preds_5 = (probs >= 0.5).astype(int)
    tn_5, fp_5, fn_5, tp_5 = confusion_matrix(labels, preds_5, labels=[0, 1]).ravel()
    acc_5 = accuracy_score(labels, preds_5)
    sens_5 = tp_5 / (tp_5 + fn_5) if (tp_5 + fn_5) > 0 else 0.0
    spec_5 = tn_5 / (tn_5 + fp_5) if (tn_5 + fp_5) > 0 else 0.0

    return {
        "auc": float(auc),
        "best_threshold": float(best_threshold),
        "accuracy": float(acc_b),
        "sensitivity": float(sens_b),
        "specificity": float(spec_b),
        # 以下字段仅用于打印参考，不在CSV留存防止混淆
        "_acc_05": float(acc_5),
        "_sens_05": float(sens_5),
        "_spec_05": float(spec_5)
    }
    # ================================================================


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到目标模型权重: {MODEL_PATH}")

    df = pd.read_csv(TEST_CSV)
    df["filename"] = df["filename"].apply(lambda f: str(Path(TEST_IMAGE_DIR) / str(f)))
    print(f"测试集样本数: {len(df)}")

    test_dataset = TestStripDataset(
        image_paths=df["filename"].tolist(),
        labels=df["label"].tolist(),
        transforms=get_val_transforms(),
        projection_transforms=get_projection_transforms(),
        standard_dpi_size=tuple(data_cfg["standard_dpi_size"]),
        proj_length=data_cfg["proj_length"],
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DualStreamStripNet(
        pretrained_2d=model_cfg["pretrained"],
        feature_dim=model_cfg["feature_dim"],
        dropout_rate=model_cfg["dropout_rate"],
    )
    model = load_model_state(model, str(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    probs = []
    with torch.no_grad():
        for img_batch, proj_batch, _ in test_loader:
            img_batch = img_batch.to(DEVICE)
            proj_batch = proj_batch.to(DEVICE)
            logits = model(img_batch, proj_batch)
            prob = torch.sigmoid(logits).cpu().numpy().flatten()
            probs.extend(prob)

    probs = np.asarray(probs, dtype=np.float32)
    labels = df["label"].astype(int).to_numpy()

    metrics = compute_metrics_with_best_threshold(labels, probs)
    metrics["model_path"] = str(MODEL_PATH)
    
    # 构建CSV只需保留核心指标
    metrics_columns = ["model_path", "best_threshold", "auc", "accuracy", "sensitivity", "specificity"]
    metrics_df = pd.DataFrame([{k: metrics[k] for k in metrics_columns}])
    metrics_csv_path = OUTPUT_DIR / "finetuned_classifier_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    df_out = df.copy()
    df_out["label"] = labels
    df_out["probability"] = probs
    df_out["pred_label"] = (probs >= metrics["best_threshold"]).astype(int)
    pred_csv_path = OUTPUT_DIR / "finetuned_classifier_predictions.csv"
    df_out.to_csv(pred_csv_path, index=False)

    print("=" * 72)
    print(f"评估模型: {MODEL_PATH}")
    print(f"整体 AUC={metrics['auc']:.4f}\n")
    print(f"[在最佳阈值 {metrics['best_threshold']:.4f} 下]")
    print(f"  Accuracy={metrics['accuracy']:.4f}")
    print(f"  Sensitivity={metrics['sensitivity']:.4f}")
    print(f"  Specificity={metrics['specificity']:.4f}\n")
    print(f"[在标准阈值 0.5000 下]")
    print(f"  Accuracy={metrics['_acc_05']:.4f}")
    print(f"  Sensitivity={metrics['_sens_05']:.4f}")
    print(f"  Specificity={metrics['_spec_05']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()