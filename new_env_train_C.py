"""
=====================================================================
new_env_train_C.py —— 纯图像流训练 (去投影流) + 图像流辅助增强模块(可开关) 启动文件
=====================================================================
范式参照 new_env_train_D.py, 尽可能复用其数据/训练/评估代码 (helper):
  - 数据读取/装载 : load_dataset_frames / build_train_val_loaders (skip_projection=True)
  - 训练流程      : 两阶段 (Stage1 冻结图像流; Stage2 余弦退火+逐层解冻),
                    早停监控 val_auc, AMP 混合精度, bootstrap 阈值
  - 输出内容      : 与 new_env_train 完全一致, 保存到 outputs/checkpoints/C/{id}/
                    (best_model.pth + history.csv + metrics.csv) 及
                    C/training_summary.csv

区别 (C 分支):
  - 【去除投影流】模型不再融合 1D 投影特征 (见 src/models/image_stream_only.py):
    仅 ImageStream 2D 特征 -> 单流分类头; 训练/验证数据加载亦跳过 1D 投影提取。
  - 【辅助增强模块可开关】在图像流特征后接轻量条带位置辅助头, 配置区 USE_AUX:
      * USE_AUX = True  (默认) : 总损失 = 主分类 BCE + AUX_LAMBDA * 条带位置辅助损失
                                  (弱标签/损失函数见 src/models/image_stream_assistance.py)
      * USE_AUX = False        : 纯图像流基线 (不做任何辅助监督)
  - 模型 (src/models/image_stream_only.py):
      ImageOnlyBase (主分类) / ImageOnlyAuxNet(base, position_head) (组合, 可开关)
  - 权重保存 (与 D 一致): best_model.pth 存主网络 base (不含辅助头);
    开启辅助时同步保存 position_head.pth (该 epoch 辅助头单独存档)。

用法 (同 new_env_train / new_env_train_D):
python new_env_train_C.py --val-csv "data/labels/val/labels.csv" --val-raw-dir "data/raw/val" --seeds "42,831,2026,7,2691" --workers 8
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
    youden_threshold,
)
from src.evaluation.metrics import compute_metrics
from src.models.image_stream_only import ImageOnlyAuxNet, ImageOnlyBase
from src.models.image_stream_assistance import (
    StripPositionHead,
    make_strip_pseudo_labels,
    strip_position_loss,
)
from src.training.train_speed_up import (
    make_amp_scaler,
    make_epoch_tqdm,
    should_validate,
)
from src.training.train_utils import get_optimizer, get_weighted_bce_loss

# ===================== 配置区 (C 分支超参, 便于替换) =====================
USE_AUX = True               # 辅助增强模块总开关: True=启用条带位置辅助(默认);
                             # False=纯图像流基线 (无任何辅助监督, 直接消融投影流)
AUX_LAMBDA = 0.35           # 辅助定位损失权重 λ (主分类 BCE 之外; 建议 0.1~0.5)
POS_CENTER = 0.5           # 阳性条带中心 y 伪标签默认值 (高度方向归一化 0~1)
Y_JITTER = 0.02            # 阳性伪标签 y 的随机扰动幅度 (模拟条带偏移; 0=关闭)
# =========================================================================

C_DIR = CHECKPOINT_ROOT / "C"                        # 输出目录 (范式同 new_env_train)
LOG_PATH = PROJECT_ROOT / "outputs" / "logs" / "training_C.log"   # 运行日志 (覆盖)
LOG_NAME = "new_env_train"       # 与 new_env_train.setup_logging 配置的 logger 名一致


# ---------------- 自定义 AMP 训练步 (仅图像流; 可选主 BCE + λ*辅助位置损失) ----------------
def _amp_train_step_C(
    net,                 # ImageOnlyAuxNet(use_aux) 或 ImageOnlyBase(no_aux)
    optimizer,
    scaler,
    criterion,
    imgs,
    labels,              # (B,) 0/1
    use_aux: bool,
) -> tuple:
    """一次 AMP 训练步 (仅图像流): 主分类 BCE + (可选) AUX_LAMBDA*条带位置辅助损失。

    Returns:
        (cls_loss, aux_loss, total_loss)
    """
    labels_b = labels.unsqueeze(1)                       # (B, 1) 给主 BCE
    if use_aux:
        presence, y_target = make_strip_pseudo_labels(
            labels.view(-1), pos_center=POS_CENTER, jitter=Y_JITTER
        )
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
        if use_aux:
            logits, aux_logits = net(imgs)
            cls_loss = criterion(logits, labels_b)
            auxd = strip_position_loss(aux_logits, presence, y_target)
            loss = cls_loss + AUX_LAMBDA * auxd["total"]
        else:
            logits = net(imgs)
            cls_loss = criterion(logits, labels_b)
            loss = cls_loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if use_aux:
        return float(cls_loss.item()), float(auxd["total"].item()), float(loss.item())
    return float(cls_loss.item()), 0.0, float(loss.item())


def _validate_C(net, val_loader, criterion, device, use_aux: bool):
    """在验证集上计算主分类 loss 与预测概率 (仅用图像, 忽略占位投影)。"""
    net.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, _projs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = net(imgs) if not use_aux else net(imgs)[0]
            loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    return val_loss, np.array(all_preds).flatten(), np.array(all_labels).flatten()


# ---------------- 训练单模型 (两阶段; 仅图像流 + 可选辅助损失) ----------------
def train_one_model_C(
    model_id: int,
    seed: int,
    train_loader,
    val_loader,
    val_labels,
    device: torch.device,
    cfg: dict,
    args,
) -> tuple:
    """训练单个 C 模型, 记录每个 epoch 验证指标 (同 new_env_train_D 的范式)。

    Returns:
        (history, metrics_row)
    """
    logger = logging.getLogger(LOG_NAME)
    model_dir = C_DIR / str(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / BEST_WEIGHT_NAME

    set_seed(seed)
    logger.info(f"\n{'=' * 70}\n>>> 训练 C 模型 {model_id} (随机种子 {seed}) | "
                f"USE_AUX={USE_AUX} AUX_LAMBDA={AUX_LAMBDA}\n{'=' * 70}")

    # 动态 pos_weight (训练集, 复用 helper)
    pos_weight, pos_cnt, neg_cnt = calc_pos_weight(train_loader)
    logger.info(f"  训练集 {pos_cnt + neg_cnt} 张 | 阳性 {pos_cnt} / 阴性 {neg_cnt} | "
                f"pos_weight={pos_weight:.3f} | 辅助增强 {'开' if USE_AUX else '关'}")

    # 主网络 base (仅图像流, 去投影) + 可选辅助头 (组合, 权重保存仅存 base)
    base = ImageOnlyBase(
        pretrained_2d=bool(cfg['model']['pretrained']),
        feature_dim=int(cfg['model']['feature_dim']),
        dropout_rate=float(cfg['model']['dropout_rate']),
    ).to(device)
    head = StripPositionHead(feature_dim=int(cfg['model']['feature_dim'])).to(device) \
        if USE_AUX else None
    net = ImageOnlyAuxNet(base, head) if USE_AUX else base
    criterion = get_weighted_bce_loss(pos_weight, device)

    patience = args.patience
    history: list = []
    best_val_auc = float('-inf')
    patience_counter = 0
    best_epoch = 0

    # ============ 阶段一: 冻结 2D 图像流 backbone (训练 fc + 分类头 + 可选辅助头) ============
    base.freeze_image_stream()
    opt1 = get_optimizer(net, lr=args.lr1, weight_decay=float(cfg['training']['weight_decay']))
    best_val_auc, patience_counter, best_epoch, history = _run_epochs_C(
        base, net, opt1, criterion, train_loader, val_loader, val_labels,
        device, epochs=args.stage1_epochs, stage_name="Stage1",
        patience=patience, model_id=model_id, seed=seed,
        history=history, save_path=save_path,
        best_val_auc=best_val_auc, patience_counter=patience_counter, best_epoch=best_epoch,
        val_interval=args.val_interval,
    )

    # ============ 阶段二: 逐层解冻 + 余弦退火 (复用 helper; param=net, backbone=base) ============
    opt2, sched2, unfreezer = build_stage2_objects(
        cfg, net, backbone_model=base,
        lr2=args.lr2,
        weight_decay=float(cfg['training']['weight_decay']),
        stage2_epochs=args.stage2_epochs,
    )
    best_val_auc, patience_counter, best_epoch, history = _run_epochs_C(
        base, net, opt2, criterion, train_loader, val_loader, val_labels,
        device, epochs=args.stage2_epochs, stage_name="Stage2",
        patience=patience, model_id=model_id, seed=seed,
        history=history, save_path=save_path,
        best_val_auc=best_val_auc, patience_counter=0, best_epoch=best_epoch,
        scheduler=sched2, unfreezer=unfreezer, val_interval=args.val_interval,
    )
    unfreeze_schedule = unfreezer.schedule_str()

    # 恢复最佳权重 (base + 可选辅助头) 后计算最终验证指标
    base.load_state_dict(torch.load(save_path, map_location="cpu"))
    head_path = model_dir / "position_head.pth"
    if USE_AUX and head_path.exists():
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
    final_loss, final_preds, final_targets = _validate_C(base, val_loader, criterion, device,
                                                         use_aux=False)
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
        "use_aux": int(USE_AUX),
        "aux_lambda": float(AUX_LAMBDA) if USE_AUX else 0.0,
        "unfreeze_schedule": unfreeze_schedule,
        "weight_path": str(save_path),
        "position_head_path": str(head_path) if (USE_AUX and head_path.exists()) else "",
    }
    import pandas as pd
    pd.DataFrame([metrics_row]).to_csv(model_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(model_dir / "history.csv", index=False)

    logger.info(f"  [C 模型 {model_id}] 最终验证 (最佳权重): AUC={m05['auc']:.4f} | "
                f"@0.5 Sens={m05['sensitivity']:.4f} Spec={m05['specificity']:.4f}")
    logger.info(f"  [C 模型 {model_id}] 约登阈值={youden_th:.4f} | bootstrap 中位数阈值={best_th:.4f} "
                f"(95%CI [{th_ci_low:.4f}, {th_ci_high:.4f}]) | "
                f"@best Sens={m_th['sensitivity']:.4f} Spec={m_th['specificity']:.4f}")
    logger.info(f"  已保存: {save_path} | history.csv | metrics.csv")
    return history, metrics_row


def _run_epochs_C(
    base, net, optimizer, criterion, train_loader, val_loader, val_labels,
    device, epochs, stage_name, patience, model_id, seed, history, save_path,
    best_val_auc, patience_counter, best_epoch, scheduler=None, unfreezer=None,
    val_interval: int = 1,
):
    """执行 C 训练循环 (每 epoch 仅用图像训练; 可选辅助损失; 验证仅用主分类 logits)。"""
    logger = logging.getLogger(LOG_NAME)
    scaler = make_amp_scaler(device)
    epoch_pbar = make_epoch_tqdm(
        range(1, epochs + 1), desc=f"[C模型{model_id}/{seed} {stage_name}]",
        leave=True, dynamic_ncols=True,
    )
    val_interval = max(1, int(val_interval))
    for epoch in epoch_pbar:
        # 0. (Stage2) 逐层解冻
        newly_unfreezed = []
        if unfreezer is not None:
            newly_unfreezed = unfreezer.apply_for_epoch(epoch, stage_name)
            if newly_unfreezed:
                msg = (f"  🔓 [C模型{model_id} {stage_name}] Epoch {epoch}: "
                       f"解冻层 {', '.join(newly_unfreezed)} "
                       f"(当前已解冻: {', '.join(unfreezer.active_layers)})")
                epoch_pbar.write(msg)
                logger.info(msg)

        # 1. 训练一轮 (AMP; 仅图像; 主分类 + 可选辅助位置损失)
        net.train()
        total_loss = 0.0
        total_aux = 0.0
        for imgs, _projs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float()
            cls_l, aux_l, tot_l = _amp_train_step_C(
                net, optimizer, scaler, criterion, imgs, labels, USE_AUX
            )
            total_loss += tot_l
            total_aux += aux_l
        train_loss = total_loss / len(train_loader)
        avg_aux = total_aux / len(train_loader)

        # 2. 验证 (主分类) + 早停判断
        do_val = should_validate(epoch, val_interval)
        if do_val:
            val_loss, preds, targets = _validate_C(base, val_loader, criterion, device,
                                                   use_aux=False)
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
                # 开启辅助时: 同步保存该最佳 epoch 的图像流辅助头 (供单独使用/复现)
                if USE_AUX:
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
            epoch_pbar.write(f"  ⚠️ [C模型{model_id} {stage_name}] 触发早停 (Epoch {epoch}/{epochs})")
            break

    return best_val_auc, patience_counter, best_epoch, history


def main():
    parser = argparse.ArgumentParser(
        description="C 分支训练: 仅图像流 (去投影流) 分类 + 可选条带位置辅助 (范式同 new_env_train)"
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
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logging(LOG_PATH)                 # 覆盖写入 training_C.log
    logger.setLevel(logging.INFO)
    logger.info("=" * 70)
    logger.info("new_env_train_C.py 启动 | 输出目录: %s", C_DIR)
    logger.info(f"配置区: USE_AUX={USE_AUX} | AUX_LAMBDA={AUX_LAMBDA} | "
                f"POS_CENTER={POS_CENTER} | Y_JITTER={Y_JITTER}")
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
    num_workers = args.workers if args.workers is not None else int(cfg['system'].get('num_workers', 0))
    logger.info("[*] 设备: %s | C 分支: 仅图像流 (skip_projection=True), 不提取 1D 投影", device)

    # 读取训练集/验证集表 + 构建 DataLoader + 默认超参 (复用 helper; 跳过投影)
    train_df, val_df, val_raw_dir_path = load_dataset_frames(
        cfg, PROJECT_ROOT, args.val_csv, val_raw_dir
    )
    logger.info("[*] 训练集: %d 张 | 验证集: %d 张 | CSV: %s",
                len(train_df), len(val_df), args.val_csv)
    train_loader, val_loader, val_labels = build_train_val_loaders(
        cfg, train_df, val_df, val_raw_dir_path, device,
        batch_size=args.batch_size, val_batch_size=args.val_batch_size,
        num_workers=num_workers, proj_cache=None, skip_projection=True,
    )
    logger.info("  验证集阳性 %d / 阴性 %d", int((val_labels == 1).sum()), int((val_labels == 0).sum()))
    resolve_hyperparams(cfg, args)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    logger.info("[*] 训练 %d 个 C 模型 | 随机种子: %s | 输出: %s", len(seeds), seeds, C_DIR)

    import pandas as pd
    all_metrics = []
    for model_id, seed in enumerate(seeds, start=1):
        history, metrics_row = train_one_model_C(
            model_id, seed, train_loader, val_loader, val_labels, device, cfg, args
        )
        metrics_row["epochs"] = len(history)
        all_metrics.append(metrics_row)

    C_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_metrics)
    summary_csv = C_DIR / "training_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info("=" * 70)
    logger.info(">>> C 训练完成 | 汇总: %s", summary_csv)
    for _, r in summary_df.iterrows():
        logger.info(
            "  C模型 %d (seed %d): best_epoch=%d | best_val_auc=%.4f | use_aux=%d λ=%.3f "
            "| best_threshold=%.4f | th_Sens=%.4f th_Spec=%.4f",
            int(r["model_id"]), int(r["seed"]), int(r["best_epoch"]),
            float(r["best_val_auc"]), int(r.get("use_aux", int(USE_AUX))),
            float(r.get("aux_lambda", AUX_LAMBDA if USE_AUX else 0.0)),
            float(r["best_threshold"]),
            float(r["th_sensitivity"]), float(r["th_specificity"]),
        )
        ufs = r.get("unfreeze_schedule", "")
        if isinstance(ufs, str) and ufs:
            logger.info("         Stage2 逐层解冻计划: %s", ufs)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
