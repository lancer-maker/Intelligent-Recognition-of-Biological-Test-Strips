"""
=====================================================================
new_env_train_D.py —— 双分支训练 + 图像流条带位置辅助模块 启动文件
=====================================================================
范式参照 new_env_train.py, 尽可能复用其数据/训练/评估代码:
  - 数据读取/装载 : read_csv_directly / prepare_labels / build_dataloader
  - 训练流程      : 两阶段 (Stage1 冻结图像流; Stage2 余弦退火+逐层解冻),
                    早停监控 val_auc, AMP 混合精度, bootstrap 阈值
  - 输出内容      : 与 new_env_train 完全一致, 保存到 outputs/checkpoints/D/{id}/
                    (best_model.pth + history.csv + metrics.csv) 及
                    D/training_summary.csv; best_model.pth 仍为裸
                    DualStreamStripNet state_dict (不含辅助头), 与
                    new_env_test / GradCAM 等加载兼容。

区别 (D 分支):
  - 主分类损失 BCE 之外, 增加"图像流条带位置辅助损失"
    (见 src/models/image_stream_assistance.py):
      总损失 = 主分类 BCE + AUX_LAMBDA * (presence BCE + center MSE)
  - 弱标签自动生成: 阳性条带中心 ≈ 高度中心(pos_center), 可加随机扰动(jitter);
                    阴性只监督"无条带"(presence=0), 忽略位置回归。
  - AUX_LAMBDA 等超参在下方"配置区"设置, 便于替换。

用法 (同 new_env_train):
python new_env_train_D.py --val-csv "data/labels/val/labels.csv" --val-raw-dir "data/raw/val" --seeds "42,831,2026,7,2691" --workers 8 --proj-cache-dir "outputs/proj_cache"
=====================================================================
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# ---------- 复用常量(new_env_train) + 共享功能(helper, 尽量复用) ----------
from new_env_train import (          # noqa: E402
    BEST_WEIGHT_NAME,
    BOOTSTRAP_SEED,
    CHECKPOINT_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    N_BOOTSTRAPS,
)
from src.utils.new_env_train_helper import (   # noqa: E402
    bootstrap_median_threshold,
    build_stage2_objects,
    build_train_val_loaders,
    calc_pos_weight,
    load_dataset_frames,
    resolve_hyperparams,
    set_seed,
    setup_logging,
    validate,
    youden_threshold,
)
from src.evaluation.metrics import compute_metrics
from src.models.dual_stream_net import DualStreamStripNet
from src.models.image_stream_assistance import (
    ImageStreamAuxNet,
    StripPositionHead,
    make_strip_pseudo_labels,
    strip_position_loss,
)
from src.training.train_speed_up import (
    ProjectionCache,
    make_amp_scaler,
    make_epoch_tqdm,
    should_validate,
)
from src.training.train_utils import get_optimizer, get_weighted_bce_loss
from src.utils.stage2_utils import (
    ProgressiveUnfreezeScheduler,
    create_stage2_optimizer,
    freeze_all_backbone,
)

# ===================== 配置区 (图像流辅助超参, 便于替换) =====================
AUX_LAMBDA = 0.35           # 辅助定位损失权重 λ (主分类 BCE 之外; 建议 0.1~0.5)
POS_CENTER = 0.5           # 阳性条带中心 y 伪标签默认值 (高度方向归一化 0~1)
Y_JITTER = 0.02            # 阳性伪标签 y 的随机扰动幅度 (模拟条带偏移; 0=关闭)
# =========================================================================

D_DIR = CHECKPOINT_ROOT / "D"                        # 输出目录 (范式同 new_env_train)
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "training_D.log"   # 运行日志 (覆盖)
LOG_NAME = "new_env_train"       # 与 new_env_train.setup_logging 配置的 logger 名一致


# ---------------- 自定义 AMP 训练步 (主 BCE + λ*辅助位置损失) ----------------
def _amp_train_step_D(
    net: ImageStreamAuxNet,
    optimizer,
    scaler,
    criterion,
    imgs,
    projs,
    labels,          # (B,) 0/1
) -> tuple:
    """一次 AMP 训练步: 主分类 BCE + AUX_LAMBDA * 条带位置辅助损失。

    Returns:
        (cls_loss, aux_loss, total_loss)
    """
    labels = labels.unsqueeze(1)                       # (B, 1) 给主 BCE
    presence, y_target = make_strip_pseudo_labels(
        labels.view(-1), pos_center=POS_CENTER, jitter=Y_JITTER
    )
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
        logits, aux_logits = net(imgs, projs)
        cls_loss = criterion(logits, labels)
        auxd = strip_position_loss(aux_logits, presence, y_target)
        loss = cls_loss + AUX_LAMBDA * auxd["total"]
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(cls_loss.item()), float(auxd["total"].item()), float(loss.item())


# ---------------- 训练单模型 (两阶段 + 辅助损失) ----------------
def train_one_model_D(
    model_id: int,
    seed: int,
    train_loader,
    val_loader,
    val_labels,
    device: torch.device,
    cfg: dict,
    args,
) -> tuple:
    """训练单个 D 模型, 记录每个 epoch 验证指标 (同 new_env_train 的范式)。

    Returns:
        (history, metrics_row)
    """
    logger = logging.getLogger(LOG_NAME)
    model_dir = D_DIR / str(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / BEST_WEIGHT_NAME

    set_seed(seed)
    logger.info(f"\n{'=' * 70}\n>>> 训练 D 模型 {model_id} (随机种子 {seed})\n{'=' * 70}")

    # 动态 pos_weight (训练集, 复用 helper)
    pos_weight, pos_cnt, neg_cnt = calc_pos_weight(train_loader)
    logger.info(f"  训练集 {pos_cnt + neg_cnt} 张 | 阳性 {pos_cnt} / 阴性 {neg_cnt} | "
                f"pos_weight={pos_weight:.3f} | AUX_LAMBDA={AUX_LAMBDA}")

    # 基模型 + 图像流辅助头 (组合, 权重保存仅存 base)
    base = DualStreamStripNet(
        pretrained_2d=bool(cfg['model']['pretrained']),
        feature_dim=int(cfg['model']['feature_dim']),
        dropout_rate=float(cfg['model']['dropout_rate']),
    ).to(device)
    head = StripPositionHead(feature_dim=int(cfg['model']['feature_dim'])).to(device)
    net = ImageStreamAuxNet(base, head)
    criterion = get_weighted_bce_loss(pos_weight, device)

    patience = args.patience
    history: list = []
    best_val_auc = float('-inf')
    patience_counter = 0
    best_epoch = 0

    # ============ 阶段一: 冻结 2D 图像流 (训练投影流+分类头+辅助头) ============
    base.freeze_image_stream()
    opt1 = get_optimizer(net, lr=args.lr1, weight_decay=float(cfg['training']['weight_decay']))
    best_val_auc, patience_counter, best_epoch, history = _run_epochs_D(
        base, net, opt1, criterion, train_loader, val_loader, val_labels,
        device, epochs=args.stage1_epochs, stage_name="Stage1",
        patience=patience, model_id=model_id, seed=seed,
        history=history, save_path=save_path,
        best_val_auc=best_val_auc, patience_counter=patience_counter, best_epoch=best_epoch,
        val_interval=args.val_interval,
    )

    # ============ 阶段二: 逐层解冻 + 余弦退火 (复用 helper; param=net 含辅助头, backbone=base) ============
    opt2, sched2, unfreezer = build_stage2_objects(
        cfg, net, backbone_model=base,
        lr2=args.lr2,
        weight_decay=float(cfg['training']['weight_decay']),
        stage2_epochs=args.stage2_epochs,
    )
    best_val_auc, patience_counter, best_epoch, history = _run_epochs_D(
        base, net, opt2, criterion, train_loader, val_loader, val_labels,
        device, epochs=args.stage2_epochs, stage_name="Stage2",
        patience=patience, model_id=model_id, seed=seed,
        history=history, save_path=save_path,
        best_val_auc=best_val_auc, patience_counter=0, best_epoch=best_epoch,
        scheduler=sched2, unfreezer=unfreezer, val_interval=args.val_interval,
    )
    unfreeze_schedule = unfreezer.schedule_str()

    # 恢复最佳权重 (base + 辅助头) 后计算最终验证指标
    base.load_state_dict(torch.load(save_path, map_location="cpu"))
    head_path = model_dir / "position_head.pth"
    if head_path.exists():
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
    final_loss, final_preds, final_targets = validate(base, val_loader, criterion, device)
    m05 = compute_metrics(final_targets, final_preds, threshold=0.5)

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
        "best_val_loss": float(min(r["val_loss"] for r in history if r["saved"])
                               if any(r["saved"] for r in history) else float('nan')),
        "final_val_loss": float(final_loss),
        "val_sensitivity": float(m05["sensitivity"]),
        "val_specificity": float(m05["specificity"]),
        "val_accuracy": float(m05["accuracy"]),
        "val_auc": float(m05["auc"]),
        "youden_threshold": float(youden_th),
        "best_threshold": float(best_th),
        "threshold_ci_low": float(th_ci_low),
        "threshold_ci_high": float(th_ci_high),
        "th_sensitivity": float(m_th["sensitivity"]),
        "th_specificity": float(m_th["specificity"]),
        "th_accuracy": float(m_th["accuracy"]),
        "aux_lambda": float(AUX_LAMBDA),
        "unfreeze_schedule": unfreeze_schedule,
        "weight_path": str(save_path),
        "position_head_path": str(head_path) if head_path.exists() else "",
    }
    import pandas as pd
    pd.DataFrame([metrics_row]).to_csv(model_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)

    logger.info(f"  [D 模型 {model_id}] 最终验证 (最佳权重): AUC={m05['auc']:.4f} | "
                f"@0.5 Sens={m05['sensitivity']:.4f} Spec={m05['specificity']:.4f}")
    logger.info(f"  [D 模型 {model_id}] 约登阈值={youden_th:.4f} | bootstrap 中位数阈值={best_th:.4f} "
                f"(95%CI [{th_ci_low:.4f}, {th_ci_high:.4f}]) | "
                f"@best Sens={m_th['sensitivity']:.4f} Spec={m_th['specificity']:.4f}")
    logger.info(f"  已保存: {save_path} | history.csv | metrics.csv")
    return history, metrics_row


def np_train_labels(train_loader):
    """取训练集标签数组 (兼容 dataset 含 .labels 属性)。"""
    return np.asarray(train_loader.dataset.labels)


def _run_epochs_D(
    base, net, optimizer, criterion, train_loader, val_loader, val_labels,
    device, epochs, stage_name, patience, model_id, seed, history, save_path,
    best_val_auc, patience_counter, best_epoch, scheduler=None, unfreezer=None,
    val_interval: int = 1,
):
    """执行 D 训练循环 (每 epoch 训练含辅助损失; 验证仅用主分类 logits)。"""
    logger = logging.getLogger(LOG_NAME)
    scaler = make_amp_scaler(device)
    epoch_pbar = make_epoch_tqdm(
        range(1, epochs + 1), desc=f"[D模型{model_id}/{seed} {stage_name}]",
        leave=True, dynamic_ncols=True,
    )
    val_interval = max(1, int(val_interval))
    for epoch in epoch_pbar:
        # 0. (Stage2) 逐层解冻
        newly_unfreezed = []
        if unfreezer is not None:
            newly_unfreezed = unfreezer.apply_for_epoch(epoch, stage_name)
            if newly_unfreezed:
                msg = (f"  🔓 [D模型{model_id} {stage_name}] Epoch {epoch}: "
                       f"解冻层 {', '.join(newly_unfreezed)} "
                       f"(当前已解冻: {', '.join(unfreezer.active_layers)})")
                epoch_pbar.write(msg)
                logger.info(msg)

        # 1. 训练一轮 (AMP + 主分类 + 辅助位置损失)
        net.train()
        total_loss = 0.0
        total_aux = 0.0
        for imgs, projs, labels in train_loader:
            imgs, projs = imgs.to(device), projs.to(device)
            labels = labels.to(device).float()
            cls_l, aux_l, tot_l = _amp_train_step_D(
                net, optimizer, scaler, criterion, imgs, projs, labels
            )
            total_loss += tot_l
            total_aux += aux_l
        train_loss = total_loss / len(train_loader)
        avg_aux = total_aux / len(train_loader)

        # 2. 验证 (主分类) + 早停判断
        do_val = should_validate(epoch, val_interval)
        if do_val:
            val_loss, preds, targets = validate(base, val_loader, criterion, device)
            m = compute_metrics(targets, preds, threshold=0.5)
            auc = float(m["auc"])
            row = {
                "model_id": model_id, "seed": seed, "stage": stage_name, "epoch": epoch,
                "train_loss": float(train_loss), "val_loss": float(val_loss),
                "val_sensitivity": float(m["sensitivity"]),
                "val_specificity": float(m["specificity"]),
                "val_accuracy": float(m["accuracy"]),
                "val_auc": auc,
                "aux_loss": float(avg_aux),
                "unfreezed_layers": ";".join(unfreezer.active_layers) if unfreezer is not None else "",
            }
            if auc > best_val_auc + 1e-4:
                best_val_auc, best_epoch, patience_counter = auc, epoch, 0
                row["saved"] = True
                torch.save(base.state_dict(), str(save_path))
                # 同步保存该最佳 epoch 的图像流辅助头 (供单独使用/复现)
                torch.save(net.position_head.state_dict(),
                           str(Path(save_path).parent / "position_head.pth"))
            else:
                row["saved"] = False
                patience_counter += 1
        else:
            auc = None
            row = {
                "model_id": model_id, "seed": seed, "stage": stage_name, "epoch": epoch,
                "train_loss": float(train_loss), "val_loss": float('nan'),
                "val_sensitivity": float('nan'), "val_specificity": float('nan'),
                "val_accuracy": float('nan'), "val_auc": float('nan'),
                "aux_loss": float(avg_aux),
                "unfreezed_layers": ";".join(unfreezer.active_layers) if unfreezer is not None else "",
                "saved": False,
            }
        history.append(row)

        if scheduler is not None:
            scheduler.step()

        postfix = {
            "t_loss": f"{train_loss:.4f}", "aux": f"{avg_aux:.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "pat": f"{patience_counter}/{patience}" if patience else "-",
        }
        if do_val:
            postfix.update({
                "v_loss": f"{val_loss:.4f}", "auc": f"{auc:.4f}",
                "sens": f"{m['sensitivity']:.3f}", "spec": f"{m['specificity']:.3f}",
            })
        epoch_pbar.set_postfix(postfix)

        if patience is not None and patience_counter >= patience:
            epoch_pbar.write(f"  ⚠️ [D模型{model_id} {stage_name}] 触发早停 (Epoch {epoch}/{epochs})")
            break

    return best_val_auc, patience_counter, best_epoch, history


def main():
    parser = argparse.ArgumentParser(
        description="D 分支训练: 双分支分类 + 图像流条带位置辅助 (范式同 new_env_train)"
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--val-csv", type=str, default="")
    parser.add_argument("--val-raw-dir", type=str, default="")
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--lr1", type=float, default=None)
    parser.add_argument("--lr2", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--proj-cache-dir", type=str, default="")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logging(LOG_PATH)                 # 覆盖写入 training_D.log
    logger.setLevel(logging.INFO)
    logger.info("=" * 70)
    logger.info("new_env_train_D.py 启动 | 输出目录: %s", D_DIR)
    logger.info(f"配置区: AUX_LAMBDA={AUX_LAMBDA} | POS_CENTER={POS_CENTER} | Y_JITTER={Y_JITTER}")
    logger.info("=" * 70)

    if not args.val_csv:
        logger.error("请通过 --val-csv 指定验证集标注 CSV。")
        return
    if not Path(args.val_csv).exists() and not Path(str(args.val_csv) + ".csv").exists():
        logger.error("验证集 CSV 不存在: %s", args.val_csv)
        return
    val_raw_dir = Path(args.val_raw_dir) if args.val_raw_dir else PROJECT_ROOT / "data" / "raw"
    if not val_raw_dir.exists():
        logger.error("验证集 raw 目录不存在: %s", val_raw_dir)
        return

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg['system']['device'] if torch.cuda.is_available() else "cpu")
    standard_dpi_size = tuple(cfg['data']['standard_dpi_size'])
    proj_length = int(cfg['data']['proj_length'])
    num_workers = args.workers if args.workers is not None else int(cfg['system'].get('num_workers', 0))
    logger.info("[*] 设备: %s", device)

    proj_cache = None
    if args.proj_cache_dir:
        proj_cache = ProjectionCache(cache_dir=args.proj_cache_dir, enabled=True)
        logger.info("[*] 投影缓存已启用: %s", args.proj_cache_dir)

    # 读取训练集/验证集表 + 构建 DataLoader + 默认超参 (复用 helper)
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
    resolve_hyperparams(cfg, args)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    logger.info("[*] 训练 %d 个 D 模型 | 随机种子: %s | 输出: %s", len(seeds), seeds, D_DIR)

    import pandas as pd
    all_metrics = []
    for model_id, seed in enumerate(seeds, start=1):
        history, metrics_row = train_one_model_D(
            model_id, seed, train_loader, val_loader, val_labels, device, cfg, args
        )
        metrics_row["epochs"] = len(history)
        all_metrics.append(metrics_row)

    D_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_metrics)
    summary_csv = D_DIR / "training_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("=" * 70)
    logger.info(">>> D 训练完成 | 汇总: %s", summary_csv)
    for _, r in summary_df.iterrows():
        logger.info(
            "  D模型 %d (seed %d): best_epoch=%d | best_val_auc=%.4f | λ=%.3f | best_threshold=%.4f "
            "| th_Sens=%.4f th_Spec=%.4f",
            int(r["model_id"]), int(r["seed"]), int(r["best_epoch"]),
            float(r["best_val_auc"]), float(r.get("aux_lambda", AUX_LAMBDA)),
            float(r["best_threshold"]),
            float(r["th_sensitivity"]), float(r["th_specificity"]),
        )
        ufs = r.get("unfreeze_schedule", "")
        if isinstance(ufs, str) and ufs:
            logger.info("         Stage2 逐层解冻计划: %s", ufs)
    logger.info("=" * 70)


def pd_concat(df_p, df_n):
    import pandas as pd
    if df_n is not None:
        return pd.concat([df_p, df_n], ignore_index=True)
    return df_p


if __name__ == "__main__":
    main()
