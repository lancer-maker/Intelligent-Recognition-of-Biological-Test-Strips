"""
=====================================================================
极简版双流模型训练与交叉验证脚本
直接读取固定 CSV，生成指定产物: loocv_predictions.csv & summary_metrics.csv
=====================================================================
"""
import os
import sys
from pathlib import Path

import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_train_transforms, get_val_transforms, get_projection_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.training.trainer import StageTrainer
from src.evaluation.threshold_optimize import calculate_best_threshold, summarize_loocv_results
from src.utils.logger import setup_logger


def read_csv_directly(csv_path: str) -> pd.DataFrame:
    """极简读取 CSV 文件 (兼容带不带 .csv 后缀)"""
    if not os.path.exists(csv_path) and os.path.exists(csv_path + ".csv"):
        csv_path += ".csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"无法找到标注文件: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_labels(df: pd.DataFrame, image_dir: Path, require_patient_id: bool = False) -> pd.DataFrame:
    """补齐患者 ID，并将 CSV 中的文件名转换为可直接读取的路径。"""
    required_columns = {'filename', 'label'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"标注文件缺少必要列: {sorted(missing_columns)}；"
            f"实际列为: {list(df.columns)}"
        )

    prepared = df.copy()
    if 'patient_id' not in prepared.columns:
        # P01_01_1.png -> 1；匿名 N_001_1.png 不参与 LOOCV 患者划分。
        patient_ids = prepared['filename'].astype(str).str.extract(r'^P(\d+)_', expand=False)
        if require_patient_id and patient_ids.isna().any():
            invalid_files = prepared.loc[patient_ids.isna(), 'filename'].tolist()
            raise ValueError(f"无法从原始数据文件名解析 patient_id: {invalid_files}")
        prepared['patient_id'] = pd.to_numeric(patient_ids, errors='coerce').astype('Int64')

    prepared['filename'] = prepared['filename'].map(
        lambda filename: str(Path(filename).resolve())
        if Path(str(filename)).is_absolute()
        else str((image_dir / str(filename)).resolve())
    )
    return prepared


def main(config_path: str = "configs/main_config.yaml"):
    # 1. 读取 YAML 配置
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parent / config_file

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_file}")

    with config_file.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg['system'].get('seed', 42))

    # 2. 物理创建固定的输出目录
    ckpt_dir = cfg['output']['checkpoint_dir']
    report_dir = "outputs/reports"
    log_dir = cfg['output'].get('log_dir', 'outputs/logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger(os.path.join(log_dir, 'training.log'))
    logger.info("训练开始，初始化日志输出和目录。")

    # 3. 直接读取 P 和 N 数据的标注 CSV
    project_root = Path(__file__).resolve().parent
    df_p = prepare_labels(
        read_csv_directly(cfg['data']['orig_csv_path']),
        project_root / "data" / "raw" / "P-24",
        require_patient_id=True,
    )
    df_n = (
        prepare_labels(
            read_csv_directly(cfg['data']['anon_csv_path']),
            project_root / "data" / "raw" / "N-16",
        )
        if os.path.exists(cfg['data']['anon_csv_path'])
        or os.path.exists(cfg['data']['anon_csv_path'] + ".csv")
        else None
    )

    # 获取 P 数据中的独立患者 ID 列表
    patient_ids = sorted(df_p['patient_id'].unique())
    num_folds = len(patient_ids)
    logger.info("开始 LOOCV 训练，自动检测到 %s 例样本，将执行 %s 折交叉验证...", num_folds, num_folds)

    fold_summaries = []
    all_predictions = []

    # 4. LOOCV 训练主循环
    for fold_idx, val_patient_id in enumerate(patient_ids, start=1):
        # 标号形式，例如 "1/24"
        fold_info_str = f"{fold_idx}/{num_folds}"

        # 划分训练集与验证集
        val_df = df_p[df_p['patient_id'] == val_patient_id].copy()
        train_p_df = df_p[df_p['patient_id'] != val_patient_id]
        train_df = pd.concat([train_p_df, df_n], ignore_index=True) if df_n is not None else train_p_df

        # 动态 pos_weight
        pos_cnt = train_df['label'].sum()
        neg_cnt = len(train_df) - pos_cnt
        pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0

        # DataLoader
        train_ds = TestStripDataset(
            image_paths=train_df['filename'].tolist(),
            labels=train_df['label'].tolist(),
            transforms=get_train_transforms(),
            projection_transforms=get_projection_transforms(),
            standard_dpi_size=tuple(cfg['data']['standard_dpi_size']),
            proj_length=cfg['data']['proj_length']
        )
        val_ds = TestStripDataset(
            image_paths=val_df['filename'].tolist(),
            labels=val_df['label'].tolist(),
            transforms=get_val_transforms(),
            projection_transforms=get_projection_transforms(),
            standard_dpi_size=tuple(cfg['data']['standard_dpi_size']),
            proj_length=cfg['data']['proj_length']
        )

        train_loader = DataLoader(train_ds, batch_size=cfg['training']['stage1']['batch_size'], shuffle=True, num_workers=cfg['system'].get('num_workers', 0))
        val_loader = DataLoader(val_ds, batch_size=cfg['training']['stage2']['batch_size'], shuffle=False)

        # 模型与两阶段训练
        model = DualStreamStripNet(
            pretrained_2d=cfg['model']['pretrained'],
            feature_dim=cfg['model']['feature_dim'],
            dropout_rate=cfg['model']['dropout_rate']
        )

        save_fold_dir = os.path.join(ckpt_dir, f"P{val_patient_id}")
        trainer = StageTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=save_fold_dir,
            pos_weight=pos_weight
        )

        # 训练并预测
        val_preds, val_targets = trainer.fit(
            phase1_epochs=cfg['training']['stage1']['epochs'],
            phase2_epochs=cfg['training']['stage2']['epochs'],
            patience=cfg['training']['early_stopping_patience'],
            phase1_lr=cfg['training']['stage1']['learning_rate'],
            phase2_lr=cfg['training']['stage2']['learning_rate'],
            weight_decay=cfg['training']['weight_decay'],
            fold_info=fold_info_str
        )

        # 计算约登指数最佳阈值与指标
        best_th, sens, spec, acc, roc_auc = calculate_best_threshold(val_targets, val_preds)

        fold_summaries.append({
            'patient_id': val_patient_id,
            'best_threshold': best_th,
            'auc': roc_auc,
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec
        })

        # 记录每张图片的预测结果
        val_df['pred_prob'] = val_preds
        val_df['pred_label'] = (val_preds >= best_th).astype(int)
        all_predictions.append(val_df)

    # 5. 导出指定的两个产物文件
    print("\n=================== 导出评估结果 ===================")
    
    # 产物 1: loocv_predictions.csv
    df_all_preds = pd.concat(all_predictions, ignore_index=True)
    pred_csv_path = os.path.join(report_dir, "loocv_predictions.csv")
    df_all_preds.to_csv(pred_csv_path, index=False)
    print(f"📄 已保存逐张试纸预测详情: {pred_csv_path}")

    # 产物 2: summary_metrics.csv
    df_summary = summarize_loocv_results(fold_summaries)
    summary_csv_path = os.path.join(report_dir, "summary_metrics.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"📄 已保存交叉验证汇总指标: {summary_csv_path}")

    print("\n🎉 训练全流程顺利完成！")


if __name__ == "__main__":
    main()