"""
=====================================================================
new_env_train.py —— 新环境典型样本验证训练脚本
=====================================================================
借鉴 test_pipeline.py 的两阶段训练流程, 但取消 24 折 LOOCV 逻辑:

  - 训练集: 原有 P-24 + N-16 全部数据 (不再按患者留出验证, 全部用于训练)
  - 验证集: 新的 14 张典型样本 (raw 图片目录与 label CSV 路径由命令行指定,
            默认留空, 由用户填入)
  - 训练若干模型 (每个模型使用不同的随机种子, 从零重新训练;
     --seeds 传几个种子就训练/保存几个模型)
  - 每个模型在每个 epoch 都记录 14 张验证集上的 灵敏度(Sensitivity) /
    特异性(Specificity) / Loss 变化 (以及 Accuracy / AUC)
  - 早停与最佳权重保存监控 val_auc (而非 val_loss)
  - Stage2 训练策略 (src/utils/stage2_utils.py):
       * 余弦退火 CosineAnnealingLR: 学习率从 lr2 衰减到接近 0 (T_max=Stage2 轮数)
       * 逐层解冻 Progressive Unfreezing: 解冻计划由 configs/main_config.yaml 控制
         (unfreeze_interval=间隔, unfreeze_start_block=起始层,
          unfreeze_num_layers=解冻层数); 默认每 3 个 epoch 解冻一层,
         解冻的 epoch 与层在终端日志、history.csv (unfreezed_layers)
         与 training_summary.csv (unfreeze_schedule) 中标出
  - 模型按 1, 2, ..., N 编号 (与传入种子一一对应), 按原有保存方式
    (裸 state_dict 的 best_model.pth) 保存到 outputs/checkpoints/Tweak/{编号}/ 下
  - 运行日志输出到 outputs/logs/training.log (每次运行直接覆盖)
  - 最终阈值: 约登指数 (Youden) 最佳平衡 + bootstrap 1000 次重采样取中位数
    (含 95% 置信区间), 写入各模型 metrics.csv 及汇总 training_summary.csv
  - 各模型完整指标 (含阈值) 合并汇总到 outputs/checkpoints/Tweak/training_summary.csv

用法示例:
  python new_env_train.py \
      --val-csv "data/labels/val/labels.csv" \
      --val-raw-dir "data/raw/val" \
      --seeds "42,831,2026"

说明:
  - --val-csv 为必填 (新 14 张典型的标注 CSV, 需含 filename,label 两列)
  - --val-raw-dir 可选: 当 CSV 中 filename 为相对路径时, 与 raw 目录拼接;
    若 filename 已是绝对路径则可不填
=====================================================================
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_train_transforms, get_val_transforms, get_projection_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.evaluation.metrics import compute_metrics
from src.training.train_utils import get_weighted_bce_loss, get_optimizer
from src.training.train_speed_up import (
    amp_train_step,
    make_amp_scaler,
    make_dataloader,
    make_epoch_tqdm,
    ProjectionCache,
    should_validate,
)
from src.utils.stage2_utils import (
    create_stage2_optimizer,
    freeze_all_backbone,
    ProgressiveUnfreezeScheduler,
)

# ===================== 可配置常量 =====================
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
TWEAK_DIR = CHECKPOINT_ROOT / "Tweak"          # 新模型保存根目录
BEST_WEIGHT_NAME = "best_model.pth"            # 与原有保存方式一致 (裸 state_dict)
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "training.log"  # 运行日志 (每次运行直接覆盖)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "main_config.yaml"
DEFAULT_SEEDS = "42,2024,2025"                 # 默认随机种子 (传几个种子就训练/保存几个模型)
N_BOOTSTRAPS = 1000                            # 阈值 bootstrap 重采样次数
BOOTSTRAP_SEED = 42                            # bootstrap 随机种子 (固定可复现)

# 通用工具与共享训练流水线封装 (见 src/utils/new_env_train_helper.py)
from src.utils.new_env_train_helper import (   # noqa: E402
    set_seed,
    setup_logging,
    youden_threshold,
    bootstrap_median_threshold,
    read_csv_directly,
    prepare_labels,
    build_dataloader,
    validate,
    load_dataset_frames,
    build_train_val_loaders,
    resolve_hyperparams,
    calc_pos_weight,
    build_stage2_objects,
)

# (原 set_seed / setup_logging / youden_threshold / bootstrap_median_threshold /
#  read_csv_directly / prepare_labels / build_dataloader / validate 的实现已移至
#  src/utils/new_env_train_helper.py, 此处仅 import, 保证单一数据源)


def train_one_model(
    model_id: int,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_labels: np.ndarray,
    device: torch.device,
    cfg: dict,
    args,
) -> Tuple[List[dict], dict]:
    """训练单个模型 (两阶段: 冻结 2D -> 解冻微调), 记录每个 epoch 验证指标。

    - 早停与最佳权重保存监控 val_auc (而非 val_loss)
    - 训练结束后基于约登指数 + bootstrap 中位数确定最终决策阈值

    Returns:
        (history, metrics_row): 逐 epoch 记录 + 该模型最终指标(含阈值)
    """
    logger = logging.getLogger("new_env_train")
    model_dir = TWEAK_DIR / str(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / BEST_WEIGHT_NAME

    set_seed(seed)
    logger.info(f"\n{'=' * 70}\n>>> 训练模型 {model_id} (随机种子 {seed})\n{'=' * 70}")

    # 动态 pos_weight (训练集上, 复用 helper)
    pos_weight, pos_cnt, neg_cnt = calc_pos_weight(train_loader)
    logger.info(f"  训练集 {pos_cnt + neg_cnt} 张 | 阳性 {pos_cnt} / 阴性 {neg_cnt} | pos_weight={pos_weight:.3f}")

    model = DualStreamStripNet(
        pretrained_2d=bool(cfg['model']['pretrained']),
        feature_dim=int(cfg['model']['feature_dim']),
        dropout_rate=float(cfg['model']['dropout_rate']),
    ).to(device)
    criterion = get_weighted_bce_loss(pos_weight, device)

    patience = args.patience
    history: List[dict] = []
    best_val_auc = float('-inf')
    patience_counter = 0
    best_epoch = 0

    # ============ 阶段一: 冻结 2D 图像流, 训练 1D 投影流与分类头 ============
    model.freeze_image_stream()
    opt1 = get_optimizer(model, lr=args.lr1, weight_decay=float(cfg['training']['weight_decay']))
    best_val_auc, patience_counter, best_epoch, history = _run_epochs(
        model, opt1, criterion, train_loader, val_loader, val_labels,
        device, epochs=args.stage1_epochs, stage_name="Stage1",
        patience=patience, model_id=model_id, seed=seed,
        history=history, save_path=save_path,
        best_val_auc=best_val_auc, patience_counter=patience_counter, best_epoch=best_epoch,
        val_interval=args.val_interval,
    )

    # ============ 阶段二: 逐层解冻 (Progressive Unfreezing) + 余弦退火 ============
    # 早停计数每阶段独立重置; 最佳 AUC 跨阶段保留, 权重保存全局 val_auc 最高的 epoch。
    # (opt2/sched2/unfreezer 复用 helper, 读 main_config stage2 配置)
    opt2, sched2, unfreezer = build_stage2_objects(
        cfg, model,
        lr2=args.lr2,
        weight_decay=float(cfg['training']['weight_decay']),
        stage2_epochs=args.stage2_epochs,
    )
    best_val_auc, patience_counter, best_epoch, history = _run_epochs(
        model, opt2, criterion, train_loader, val_loader, val_labels,
        device, epochs=args.stage2_epochs, stage_name="Stage2",
        patience=patience, model_id=model_id, seed=seed,
        history=history, save_path=save_path,
        best_val_auc=best_val_auc, patience_counter=0, best_epoch=best_epoch,
        scheduler=sched2, unfreezer=unfreezer, val_interval=args.val_interval,
    )
    unfreeze_schedule = unfreezer.schedule_str()    # 如 'blocks.6@e2, blocks.5@e4'

    # 恢复最佳权重 (val_auc 最高 epoch) 后计算最终验证指标
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    final_loss, final_preds, final_targets = validate(model, val_loader, criterion, device)
    m05 = compute_metrics(final_targets, final_preds, threshold=0.5)

    # 阈值: 约登指数 + bootstrap 中位数 (小样本稳定化)
    youden_th = youden_threshold(final_targets, final_preds)
    best_th, th_ci_low, th_ci_high = bootstrap_median_threshold(
        final_targets, final_preds, n_bootstraps=N_BOOTSTRAPS, seed=BOOTSTRAP_SEED + model_id
    )
    m_th = compute_metrics(final_targets, final_preds, threshold=best_th)

    metrics_row = {
        "model_id": model_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "best_val_loss": float(min(r["val_loss"] for r in history if r["saved"]) if any(r["saved"] for r in history) else float('nan')),
        "final_val_loss": float(final_loss),
        # @阈值 0.5 的常规指标
        "val_sensitivity": float(m05["sensitivity"]),
        "val_specificity": float(m05["specificity"]),
        "val_accuracy": float(m05["accuracy"]),
        "val_auc": float(m05["auc"]),
        # 阈值 (约登 + bootstrap 中位数) 及其 95% 置信区间
        "youden_threshold": float(youden_th),
        "best_threshold": float(best_th),
        "threshold_ci_low": float(th_ci_low),
        "threshold_ci_high": float(th_ci_high),
        # 最佳阈值下的指标
        "th_sensitivity": float(m_th["sensitivity"]),
        "th_specificity": float(m_th["specificity"]),
        "th_accuracy": float(m_th["accuracy"]),
        # Stage2 逐层解冻计划 (如 'blocks.6@e2, blocks.5@e4')
        "unfreeze_schedule": unfreeze_schedule,
        "weight_path": str(save_path),
    }
    pd.DataFrame([metrics_row]).to_csv(model_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)

    logger.info(
        f"  [模型 {model_id}] 最终验证 (最佳权重): AUC={m05['auc']:.4f} | "
        f"@0.5 Sens={m05['sensitivity']:.4f} Spec={m05['specificity']:.4f}"
    )
    logger.info(
        f"  [模型 {model_id}] 约登阈值={youden_th:.4f} | bootstrap中位数阈值={best_th:.4f} "
        f"(95%CI [{th_ci_low:.4f}, {th_ci_high:.4f}]) | "
        f"@best Sens={m_th['sensitivity']:.4f} Spec={m_th['specificity']:.4f}"
    )
    logger.info(f"  已保存: {save_path} | history.csv | metrics.csv")
    return history, metrics_row


def _run_epochs(
    model, optimizer, criterion, train_loader, val_loader, val_labels,
    device, epochs: int, stage_name: str, patience: Optional[int],
    model_id: int, seed: int, history: List[dict], save_path: Path,
    best_val_auc: float, patience_counter: int, best_epoch: int,
    scheduler=None, unfreezer=None, val_interval: int = 1,
):
    """执行一个阶段的训练循环, 每个 epoch 记录验证集灵敏度/特异性/loss/AUC。

    早停与最佳权重保存监控 val_auc (越高越好), 不再监控 val_loss。
    - Stage2 使用逐层解冻 (unfreezer) 与余弦退火 (scheduler):
      unfreezer.apply_for_epoch() 在每个 epoch 训练前调用 (逐层解冻);
      scheduler.step() 在每个 epoch 结束后调用 (学习率衰减)。
    - 训练加速 (src/training/train_speed_up.py):
      * AMP 混合精度: 每步前向/反向经 autocast + GradScaler (仅 CUDA 生效)
      * 验证频率: 每 val_interval 个 epoch 验证一次并做早停判断 (非验证 epoch 不改变早停计数)
      * tqdm mininterval=1.0, 降低日志刷新频率
    """
    logger = logging.getLogger("new_env_train")
    scaler = make_amp_scaler(device)
    epoch_pbar = make_epoch_tqdm(
        range(1, epochs + 1), desc=f"[模型{model_id}/{seed} {stage_name}]",
        leave=True, dynamic_ncols=True,
    )
    val_interval = max(1, int(val_interval))
    for epoch in epoch_pbar:
        # 0. (Stage2) 逐层解冻: 每 interval 个 epoch 解冻一个 block (从最后往前)
        newly_unfreezed: List[str] = []
        if unfreezer is not None:
            newly_unfreezed = unfreezer.apply_for_epoch(epoch, stage_name)
            if newly_unfreezed:
                msg = (f"  🔓 [模型{model_id} {stage_name}] Epoch {epoch}: "
                       f"解冻层 {', '.join(newly_unfreezed)} "
                       f"(当前已解冻: {', '.join(unfreezer.active_layers)})")
                tqdm.write(msg)
                logger.info(msg)

        # 1. 训练一轮 (AMP 混合精度)
        model.train()
        total_loss = 0.0
        for imgs, projs, labels in train_loader:
            imgs, projs, labels = imgs.to(device), projs.to(device), labels.to(device).unsqueeze(1)
            total_loss += amp_train_step(model, optimizer, scaler, criterion, imgs, projs, labels)
        train_loss = total_loss / len(train_loader)

        # 2. 验证 (每 val_interval 个 epoch 一次) + 早停判断 (监控 val_auc)
        do_val = should_validate(epoch, val_interval)
        if do_val:
            val_loss, preds, targets = validate(model, val_loader, criterion, device)
            m = compute_metrics(targets, preds, threshold=0.5)
            auc = float(m["auc"])
            row = {
                "model_id": model_id,
                "seed": seed,
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_sensitivity": float(m["sensitivity"]),
                "val_specificity": float(m["specificity"]),
                "val_accuracy": float(m["accuracy"]),
                "val_auc": auc,
                "unfreezed_layers": ";".join(unfreezer.active_layers) if unfreezer is not None else "",
            }
            if auc > best_val_auc + 1e-4:
                best_val_auc = auc
                best_epoch = epoch
                patience_counter = 0
                row["saved"] = True
                torch.save(model.state_dict(), str(save_path))
            else:
                row["saved"] = False
                patience_counter += 1
        else:
            # 非验证 epoch: val 字段记 NaN, 不参与早停
            auc = None
            row = {
                "model_id": model_id,
                "seed": seed,
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": np.nan,
                "val_sensitivity": np.nan,
                "val_specificity": np.nan,
                "val_accuracy": np.nan,
                "val_auc": np.nan,
                "unfreezed_layers": ";".join(unfreezer.active_layers) if unfreezer is not None else "",
                "saved": False,
            }

        history.append(row)
        # (Stage2) 余弦退火: 每个 epoch 结束后更新学习率
        if scheduler is not None:
            scheduler.step()

        postfix = {
            "t_loss": f"{train_loss:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "pat": f"{patience_counter}/{patience}" if patience else "-",
        }
        if do_val:
            postfix.update({
                "v_loss": f"{val_loss:.4f}",
                "auc": f"{auc:.4f}",
                "sens": f"{m['sensitivity']:.3f}",
                "spec": f"{m['specificity']:.3f}",
            })
        epoch_pbar.set_postfix(postfix)

        if patience is not None and patience_counter >= patience:
            tqdm.write(f"  ⚠️ [模型{model_id} {stage_name}] 触发早停 (Epoch {epoch}/{epochs})")
            break

    return best_val_auc, patience_counter, best_epoch, history


def main():
    parser = argparse.ArgumentParser(
        description="新环境典型样本验证训练: 全量训练集 + 新 14 张验证集, 训练 3 个随机种子模型"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="YAML 配置文件路径")
    parser.add_argument("--val-csv", type=str, default="", help="新 14 张典型样本的标注 CSV 路径 (必填)")
    parser.add_argument("--val-raw-dir", type=str, default="", help="新 14 张典型样本的 raw 图片目录 (filename 为相对路径时必填)")
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS, help="各模型随机种子, 逗号分隔 (传几个就训练/保存几个, 默认 3 个)")
    parser.add_argument("--batch-size", type=int, default=None, help="训练 batch size (默认取配置 stage1)")
    parser.add_argument("--val-batch-size", type=int, default=8, help="验证 batch size (默认 8)")
    parser.add_argument("--stage1-epochs", type=int, default=None, help="阶段一 epochs (默认取配置)")
    parser.add_argument("--stage2-epochs", type=int, default=None, help="阶段二 epochs (默认取配置)")
    parser.add_argument("--lr1", type=float, default=None, help="阶段一学习率 (默认取配置)")
    parser.add_argument("--lr2", type=float, default=None, help="阶段二学习率 (默认取配置)")
    parser.add_argument("--patience", type=int, default=None, help="早停耐心 (默认取配置)")
    parser.add_argument("--val-interval", type=int, default=1,
                        help="每多少个 epoch 验证一次并做早停判断 (默认 1; 设 2~3 可减少验证开销)")
    parser.add_argument("--proj-cache-dir", type=str, default="",
                        help="投影缓存目录 (启用后基于原始图像缓存投影, 加速数据加载; 默认空=不启用, 不影响原 CPU 逻辑)")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader 线程数 (默认取配置)")
    args = parser.parse_args()

    # 0. 配置日志 (每次运行直接覆盖 outputs/logs/training.log)
    logger = setup_logging(LOG_PATH)
    logger.info("=" * 70)
    logger.info("new_env_train.py 启动 | 日志文件(覆盖): %s", LOG_PATH)
    logger.info("=" * 70)

    # 1. 校验验证集路径 (用户需填入新 14 张典型的 raw 与 label 路径)
    if not args.val_csv:
        logger.error("请通过 --val-csv 指定新 14 张典型的标注 CSV 路径 (raw 目录可用 --val-raw-dir 指定)。")
        return
    if not Path(args.val_csv).exists() and not Path(str(args.val_csv) + ".csv").exists():
        logger.error("验证集 CSV 不存在: %s", args.val_csv)
        return
    val_raw_dir = Path(args.val_raw_dir) if args.val_raw_dir else PROJECT_ROOT / "data" / "raw"
    if not val_raw_dir.exists():
        logger.error("验证集 raw 目录不存在: %s (filename 为绝对路径时可留空 --val-raw-dir)", val_raw_dir)
        return

    # 2. 读取配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else "cpu")
    logger.info("[*] 设备: %s | 输出目录: %s", device, TWEAK_DIR)

    standard_dpi_size = tuple(cfg['data']['standard_dpi_size'])
    proj_length = int(cfg['data']['proj_length'])
    num_workers = args.workers if args.workers is not None else int(cfg['system'].get('num_workers', 0))

    # 可选: 投影缓存 (基于原始图像, 默认不启用, 不影响原 CPU 逻辑)
    proj_cache = None
    if args.proj_cache_dir:
        proj_cache = ProjectionCache(cache_dir=args.proj_cache_dir, enabled=True)
        logger.info("[*] 投影缓存已启用: %s", args.proj_cache_dir)

    # 3+4. 读取训练集(P-24+N-16)/验证集表 + 5. 构建 DataLoader (复用 helper)
    train_df, val_df, val_raw_dir_path = load_dataset_frames(
        cfg, PROJECT_ROOT, args.val_csv, val_raw_dir
    )
    logger.info("[*] 训练集: %d 张 | 验证集: %d 张 | CSV: %s",
                len(train_df), len(val_df), args.val_csv)
    train_loader, val_loader, val_labels = build_train_val_loaders(
        cfg, train_df, val_df, val_raw_dir_path, device,
        batch_size=args.batch_size, val_batch_size=args.val_batch_size,
        num_workers=num_workers, proj_cache=proj_cache,
    )
    logger.info("  验证集阳性 %d / 阴性 %d", int((val_labels == 1).sum()), int((val_labels == 0).sum()))

    # 6. 覆盖默认训练超参数 (复用 helper)
    resolve_hyperparams(cfg, args)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    logger.info("[*] 训练 %d 个模型 | 随机种子: %s | 归档: %s", len(seeds), seeds, TWEAK_DIR)

    # 7. 依次训练每个模型
    all_metrics = []
    for model_id, seed in enumerate(seeds, start=1):
        history, metrics_row = train_one_model(model_id, seed, train_loader, val_loader,
                                               val_labels, device, cfg, args)
        metrics_row["epochs"] = len(history)
        all_metrics.append(metrics_row)

    # 8. 汇总: 各模型完整 metrics (含约登/bootstraps 最佳阈值) 合并写入 training_summary.csv
    TWEAK_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_metrics)
    summary_csv = TWEAK_DIR / "training_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("\n" + "=" * 70)
    logger.info(">>> 训练完成, 共 %d 个模型已按编号保存到 %s", len(seeds), TWEAK_DIR)
    logger.info(">>> 汇总 (含最佳阈值): %s", summary_csv)
    for _, r in summary_df.iterrows():
        logger.info(
            "  模型 %d (seed %d): best_epoch=%d | best_val_auc=%.4f | "
            "best_threshold=%.4f [95%%CI %.4f~%.4f] | th_Sens=%.4f th_Spec=%.4f",
            int(r["model_id"]), int(r["seed"]), int(r["best_epoch"]),
            float(r["best_val_auc"]), float(r["best_threshold"]),
            float(r["threshold_ci_low"]), float(r["threshold_ci_high"]),
            float(r["th_sensitivity"]), float(r["th_specificity"]),
        )
        ufs = r.get("unfreeze_schedule", "")
        if isinstance(ufs, str) and ufs:
            logger.info("         Stage2 逐层解冻计划: %s", ufs)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
