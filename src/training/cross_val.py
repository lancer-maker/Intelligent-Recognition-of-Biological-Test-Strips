# 留一患者交叉验证（LOOCV）核心逻辑控制脚本
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Optional

from src.data.dataset import TestStripDataset
from src.data.transforms import get_train_transforms, get_val_transforms, get_projection_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.training.trainer import StageTrainer

from configs.load_config import get_main_config

config = get_main_config()
system_cfg = config.get("system", {})
data_cfg = config.get("data", {})
training_cfg = config.get("training", {})
stage1_cfg = training_cfg.get("stage1", {})
stage2_cfg = training_cfg.get("stage2", {})
model_cfg = config.get("model", {})
evaluation_cfg = config.get("evaluation", {})
output_cfg = config.get("output", {})

# 假设 evaluation 模块提供该函数（若尚未实现，脚本内部会自动捕获降级）
try:
    from src.evaluation.threshold_optimize import calculate_best_threshold
except ImportError:
    from sklearn.metrics import roc_auc_score, accuracy_score

    def calculate_best_threshold(y_true, y_prob):
        auc_val = roc_auc_score(y_true, y_prob)
        threshold = float(evaluation_cfg.get("default_threshold", 0.5))
        return threshold, 0.0, 0.0, accuracy_score(y_true, y_prob >= threshold), auc_val


def run_loocv(
    orig_csv: Optional[str] = None,
    anon_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_patients: Optional[int] = None,
    device_str: Optional[str] = None
) -> pd.DataFrame:
    """
    运行 24 折留一患者交叉验证 (LOOCV)。
    
    Args:
        orig_csv: 包含原有 192 张图的 CSV 路径 (字段: filename, patient_id, label)
        anon_csv: 新增 128 张匿名图的 CSV 路径 (字段: filename, label)，仅加入训练集
        output_dir: 结果与模型保存目录
        num_patients: 患者总数，默认 24
        device_str: 设备名称 ('cuda' 或 'cpu')
    """
    orig_csv = orig_csv or data_cfg.get("orig_csv_path")
    anon_csv = anon_csv or data_cfg.get("anon_csv_path")
    output_dir = output_dir or output_cfg.get("checkpoint_dir", "outputs/checkpoints")
    num_patients = num_patients or int(evaluation_cfg.get("num_patients", 24))
    device_str = device_str or system_cfg.get("device", "cuda")

    if not orig_csv:
        raise ValueError("未提供 orig_csv，且配置文件 data.orig_csv_path 为空。")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # 读取数据
    orig_df = pd.read_csv(orig_csv)
    anon_df = pd.read_csv(anon_csv) if (anon_csv and os.path.exists(anon_csv)) else None

    fold_results = []

    for val_patient_id in range(1, num_patients + 1):
        print(f"\n==================== LOOCV Fold {val_patient_id} / {num_patients} ====================")

        # 1. 划分数据集
        val_df = orig_df[orig_df['patient_id'] == val_patient_id]
        train_orig_df = orig_df[orig_df['patient_id'] != val_patient_id]

        # 匿名数据只能加入训练集
        if anon_df is not None:
            train_df = pd.concat([train_orig_df, anon_df], ignore_index=True)
        else:
            train_df = train_orig_df

        # 2. 计算训练集的 pos_weight
        pos_cnt = train_df['label'].sum()
        neg_cnt = len(train_df) - pos_cnt
        pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0

        # 3. 创建 DataLoader
        train_ds = TestStripDataset(
            image_paths=train_df['filename'].tolist(),
            labels=train_df['label'].tolist(),
            transforms=get_train_transforms(),
            projection_transforms=get_projection_transforms()
        )
        val_ds = TestStripDataset(
            image_paths=val_df['filename'].tolist(),
            labels=val_df['label'].tolist(),
            transforms=get_val_transforms(),
            projection_transforms=get_projection_transforms()
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=int(stage1_cfg.get("batch_size", 16)),
            shuffle=True,
            num_workers=int(system_cfg.get("num_workers", 0))
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(stage2_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=int(system_cfg.get("num_workers", 0))
        )

        # 4. 初始化模型与训练器
        model = DualStreamStripNet(
            pretrained_2d=bool(model_cfg.get("pretrained", True))
        )
        fold_save_dir = os.path.join(output_dir, f"fold_{val_patient_id}")
        
        trainer = StageTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=fold_save_dir,
            pos_weight=pos_weight
        )

        # 5. 执行训练
        preds, targets = trainer.fit()

        # 6. 计算基于约登指数的最佳阈值与指标
        best_th, sens, spec, acc, roc_auc = calculate_best_threshold(targets, preds)

        fold_results.append({
            'patient_id': val_patient_id,
            'best_threshold': best_th,
            'auc': roc_auc,
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec
        })

        print(f"Fold {val_patient_id} @ Best TH({best_th:.3f}) -> AUC: {roc_auc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")

    # 汇总输出
    results_df = pd.DataFrame(fold_results)
    summary_path = os.path.join(output_dir, "loocv_summary.csv")
    results_df.to_csv(summary_path, index=False)

    print("\n==================== LOOCV Overall Summary ====================")
    print(f"Mean AUC:         {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"Mean Sensitivity: {results_df['sensitivity'].mean():.4f} ± {results_df['sensitivity'].std():.4f}")
    print(f"Mean Specificity: {results_df['specificity'].mean():.4f} ± {results_df['specificity'].std():.4f}")
    print(f"Mean Accuracy:    {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Mean Best TH:     {results_df['best_threshold'].mean():.4f}")
    print(f"Summary CSV Saved: {summary_path}")

    return results_df