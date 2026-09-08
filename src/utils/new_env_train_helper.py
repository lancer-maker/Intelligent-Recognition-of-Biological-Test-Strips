"""
=====================================================================
new_env_train_helper.py —— new_env_train / new_env_train_D 共享功能模块
=====================================================================
将两个训练启动脚本重复使用的基础工具与训练流水线片段集中于此 (单一数据源):

  A. 通用工具 (原散落在 new_env_train.py):
     - set_seed / setup_logging
     - youden_threshold / bootstrap_median_threshold
     - read_csv_directly / prepare_labels / build_dataloader / validate

  B. 共享构建器 (供两个脚本的 main / train_one_model 复用):
     - load_dataset_frames      : 读取 P-24+N-16 训练表 + 验证集表(绝对路径化)
     - build_train_val_loaders  : 依配置构建 train/val DataLoader (含增强)
     - resolve_hyperparams      : 用配置补齐 stage1/stage2 epochs·lr·patience
     - calc_pos_weight          : 按训练集正负比计算 pos_weight
     - build_stage2_objects     : 依配置生成 Stage2 (opt2,sched2,unfreezer)

用法: 两个启动脚本从本模块 import, 不再各自实现重复逻辑。
=====================================================================
"""
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_projection_transforms, get_train_transforms, get_val_transforms
from src.training.train_speed_up import make_dataloader
from src.utils.stage2_utils import (
    ProgressiveUnfreezeScheduler,
    create_stage2_optimizer,
    freeze_all_backbone,
)

# 通用默认 (脚本可覆盖)
N_BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 42


# ============================= A. 通用工具 =============================
def set_seed(seed: int) -> None:
    """固定全局随机种子, 保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_path: Path, logger_name: str = "new_env_train") -> logging.Logger:
    """配置日志: 同时输出到控制台与 log_path 文件 (mode='w' 直接覆盖)。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):          # 清空旧 handler, 避免重复
        logger.removeHandler(h)
        h.close()
    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """约登指数 (Youden Index) 最佳阈值: 在灵敏度与特异度之间取得最佳平衡。

    验证集仅单一类别时回退到 0.5。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr + (1.0 - fpr) - 1.0
    best_idx = int(np.argmax(youden))
    th = float(thresholds[best_idx])
    if np.isinf(th):                          # sklearn 边界伪影处理
        th = 1.0
    return th


def bootstrap_median_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """小样本阈值稳定化: bootstrap 重采样求约登阈值分布中位数与 95% CI。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    idxs = np.arange(n)
    thresholds = []
    for _ in range(n_bootstraps):
        idx = rng.choice(idxs, size=n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:            # 重采样后单类则跳过
            continue
        thresholds.append(youden_threshold(yt, yp))
    if not thresholds:
        return 0.5, 0.5, 0.5
    thresholds = np.asarray(thresholds)
    return (float(np.median(thresholds)),
            float(np.percentile(thresholds, 2.5)),
            float(np.percentile(thresholds, 97.5)))


def read_csv_directly(csv_path: str) -> pd.DataFrame:
    """极简读取 CSV 文件 (兼容带不带 .csv 后缀)。"""
    if not Path(csv_path).exists() and Path(str(csv_path) + ".csv").exists():
        csv_path = str(csv_path) + ".csv"
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"无法找到标注文件: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_labels(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """校验必需列, 并将 CSV 中的文件名转换为可直接读取的绝对路径。"""
    required_columns = {'filename', 'label'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"标注文件缺少必要列: {sorted(missing_columns)}；实际列为: {list(df.columns)}"
        )
    prepared = df.copy()
    prepared['filename'] = prepared['filename'].map(
        lambda filename: str(Path(filename).resolve())
        if Path(str(filename)).is_absolute()
        else str((image_dir / str(filename)).resolve())
    )
    return prepared


def build_dataloader(
    df: pd.DataFrame,
    image_dir: Path,
    transforms,
    projection_transforms,
    standard_dpi_size: Tuple[int, int],
    proj_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    proj_cache=None,
    skip_projection: bool = False,
) -> Tuple[DataLoader, np.ndarray]:
    """根据标注 DataFrame 构建 DataLoader (经 train_speed_up.make_dataloader, CUDA pin_memory)。

    skip_projection=True 时不提取 1D 投影 (纯图像流训练用, dataset 返回占位零张量)。
    """
    prepared = prepare_labels(df, image_dir)
    ds = TestStripDataset(
        image_paths=prepared['filename'].tolist(),
        labels=prepared['label'].tolist(),
        transforms=transforms,
        projection_transforms=projection_transforms,
        standard_dpi_size=standard_dpi_size,
        proj_length=proj_length,
        proj_cache=proj_cache,
        skip_projection=skip_projection,
    )
    loader = make_dataloader(ds, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, device=device)
    return loader, prepared['label'].to_numpy(dtype=int)


def validate(model, val_loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray]:
    """在验证集上计算 loss 与预测概率。"""
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, projs, labels in val_loader:
            imgs = imgs.to(device)
            projs = projs.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(imgs, projs)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    return val_loss, np.array(all_preds).flatten(), np.array(all_labels).flatten()


# ============================= B. 共享构建器 =============================
def load_dataset_frames(
    cfg: dict,
    project_root: Path,
    val_csv: str,
    val_raw_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    """读取训练表 (P-24 + N-16) 与验证表, 均转为绝对路径; 返回 (train_df, val_df, val_raw_dir)。

    Raises:
        FileNotFoundError: 验证集 CSV/raw 目录缺失时。
    """
    val_csv = str(val_csv)
    if not Path(val_csv).exists() and not Path(str(val_csv) + ".csv").exists():
        raise FileNotFoundError(f"验证集 CSV 不存在: {val_csv}")
    val_raw_dir = Path(val_raw_dir) if val_raw_dir else project_root / "data" / "raw"
    if not val_raw_dir.exists():
        raise FileNotFoundError(f"验证集 raw 目录不存在: {val_raw_dir}")

    df_p = prepare_labels(read_csv_directly(cfg['data']['orig_csv_path']),
                          project_root / "data" / "raw" / "P-24")
    anon_csv = cfg['data']['anon_csv_path']
    df_n = None
    if Path(anon_csv).exists() or Path(str(anon_csv) + ".csv").exists():
        df_n = prepare_labels(read_csv_directly(anon_csv),
                              project_root / "data" / "raw" / "N-16")
    train_df = pd.concat([df_p, df_n], ignore_index=True) if df_n is not None else df_p

    val_df = read_csv_directly(val_csv)
    if len(val_df) == 0:
        raise ValueError(f"验证集 CSV 为空 (只有表头?): {val_csv}")
    return train_df, val_df, val_raw_dir


def build_train_val_loaders(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_raw_dir: Path,
    device: torch.device,
    *,
    batch_size: Optional[int] = None,
    val_batch_size: int = 8,
    num_workers: int = 0,
    proj_cache=None,
    skip_projection: bool = False,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """按配置构建 train/val DataLoader (训练/验证增强与 new_env_train 一致)。

    skip_projection=True 时不提取 1D 投影 (纯图像流/去投影消融训练)。
    """
    standard_dpi_size = tuple(cfg['data']['standard_dpi_size'])
    proj_length = int(cfg['data']['proj_length'])
    train_batch = batch_size if batch_size else int(cfg['training']['stage1']['batch_size'])

    train_loader, _ = build_dataloader(
        train_df, PROJECT_ROOT / "data" / "raw" / "P-24",      # 路径已绝对化, 此参数仅占位
        get_train_transforms(), get_projection_transforms(),
        standard_dpi_size, proj_length,
        batch_size=train_batch, shuffle=True,
        num_workers=num_workers, device=device, proj_cache=proj_cache,
        skip_projection=skip_projection,
    )
    val_loader, val_labels = build_dataloader(
        val_df, val_raw_dir,
        get_val_transforms(), get_projection_transforms(),
        standard_dpi_size, proj_length,
        batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, device=device, proj_cache=proj_cache,
        skip_projection=skip_projection,
    )
    return train_loader, val_loader, val_labels


def resolve_hyperparams(cfg: dict, args) -> None:
    """用配置补齐 args 中未显式给出的 stage1/stage2 epochs·lr·patience。"""
    if args.stage1_epochs is None:
        args.stage1_epochs = int(cfg['training']['stage1']['epochs'])
    if args.stage2_epochs is None:
        args.stage2_epochs = int(cfg['training']['stage2']['epochs'])
    if args.lr1 is None:
        args.lr1 = float(cfg['training']['stage1']['learning_rate'])
    if args.lr2 is None:
        args.lr2 = float(cfg['training']['stage2']['learning_rate'])
    if args.patience is None:
        args.patience = cfg['training']['early_stopping_patience']


def calc_pos_weight(train_loader: DataLoader) -> Tuple[float, int, int]:
    """按训练集正负比计算 BCE pos_weight, 返回 (pos_weight, pos_cnt, neg_cnt)。"""
    train_labels = np.asarray(train_loader.dataset.labels)
    pos_cnt = int((train_labels == 1).sum())
    neg_cnt = int((train_labels == 0).sum())
    pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    return float(pos_weight), pos_cnt, neg_cnt


def build_stage2_objects(
    cfg: dict,
    param_model,
    backbone_model=None,
    *,
    lr2: float,
    weight_decay: float,
    stage2_epochs: int,
):
    """依配置创建 Stage2 (余弦退火优化器 + 逐层解冻调度器)。

    Args:
        param_model  : 提供 .parameters() 的优化对象 (new_env_train 传 DualStreamStripNet;
                       D 版可传组合模型 ImageStreamAuxNet 以包含辅助头)
        backbone_model: 含 image_stream.backbone 的解冻对象 (默认同 param_model;
                        D 版传其中的 base, 因组合模型无 .image_stream 属性)
    Returns:
        (opt2, sched2, unfreezer)
    """
    if backbone_model is None:
        backbone_model = param_model
    freeze_all_backbone(backbone_model)                # Stage2 前默认冻结全部图像流
    opt2, sched2 = create_stage2_optimizer(
        param_model, lr=lr2, weight_decay=weight_decay,
        t_max=stage2_epochs, eta_min=1e-7,
    )
    stage2_cfg = cfg['training']['stage2']
    unfreeze_interval = int(stage2_cfg.get('unfreeze_interval', 3))
    unfreeze_start = stage2_cfg.get('unfreeze_start_block')
    unfreeze_layers = stage2_cfg.get('unfreeze_num_layers')
    unfreezer = ProgressiveUnfreezeScheduler(
        backbone_model,                              # 其 image_stream.backbone 为解冻对象
        interval=unfreeze_interval,
        start_block=int(unfreeze_start) if unfreeze_start is not None else None,
        num_layers=int(unfreeze_layers) if unfreeze_layers is not None else None,
    )
    return opt2, sched2, unfreezer
