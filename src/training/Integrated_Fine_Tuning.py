"""
基于 24 个已训练模型的困难样本分类头微调、集成评估与结果归档
============================================================
参考脚本:
  - tests/finetune_classifier.py : 困难样本挖掘 + 分类头微调
  - tests/eva_fin_classifier.py  : 单模型测试集评估
  - tests/finetune_pipeline.py   : 微调-评估-归档调度

相比参考脚本的核心变化:
  1. 对 outputs/checkpoints 下 P1~P24 共 24 个已训练模型逐一微调分类头。
  2. 每个模型在其"训练侧数据"(排除其 LOOCV 验证患者) 上挖掘困难样本作为微调集,
     以其 LOOCV 验证患者作为监控验证集。
  3. 移除参考脚本中对单张样本 (N_026_1) 的监控, 改为在验证集上监控
     验证损失 (val_loss) 与 AUC, 结合早停防止微调过拟合退化。
  4. 独立测试集路径留白 (--test-csv 默认为空); 集成构建不依赖测试集, 始终执行,
     测试集仅用于可选的最终报告阶段 (有 --test-csv 时才评估)。
  5. 无测试集泄漏: 单模型筛选基于 LOOCV 验证集指标 (final_val_auc / final_val_sensitivity),
     决策阈值固定 0.5 (不在测试集上寻优), 测试集只用于计算最终报告指标。
  6. 集成双版本: 筛选版 (验证集指标达标) 与全量版 (24 个模型全用), 概率平均 (Soft Voting),
     单模型命名保持 P1~P24 序列不变。
  7. 结果归档: 每个模型的权重/训练历史/最终验证指标保存在
     outputs/reports/test/<RUN_NAME>/P{序号}/ 文件夹下;
     集成模型参数、汇总与预测保存在 run 根目录。

归档结构 (RUN_NAME 默认 "ensemble_finetune"):
  outputs/reports/test/ensemble_finetune/
    ensemble_model.pth            # 筛选版集成参数: {model_ids, weights:{'P1': state_dict,...}, best_threshold:0.5}
    ensemble_all_model.pth        # 全量版集成参数 (24 个模型全用)
    ensemble_summary.csv          # 筛选版测试集汇总 (阈值 0.5)
    ensemble_all_summary.csv      # 全量版测试集汇总 (阈值 0.5)
    ensemble_predictions.csv      # 筛选版 test_probs + test_labels
    ensemble_all_predictions.csv  # 全量版 test_probs + test_labels
    per_model_validate_metrics.csv  # 各单模型 LOOCV 验证集指标 (用于筛选, 无泄漏)
    per_model_test_metrics.csv    # 各单模型测试集指标 (阈值 0.5, 仅报告)
    P01/
      finetuned_classifier.pth    # 微调后权重 (若微调未改善验证指标则不生成, 评估回退原始权重)
      history.csv                 # epoch, val_loss, val_auc 等微调数组
      metrics.csv                 # 最终 val_auc, val_sensitivity 配对
    P02/ ...
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# ===================== 功能性代码已封装至 src/utils/Fine_Tuning_helper.py =====================
from src.utils.Fine_Tuning_helper import (
    CHECKPOINT_ROOT,
    RUN_DIR,
    FINETUNE_WEIGHT_NAME,
    ORIGINAL_WEIGHT_NAME,
    RUN_NAME,
    TEST_CSV,
    set_seed,
    load_config,
    parse_model_ids,
    discover_model_ids,
    load_training_records,
    split_records_by_patient,
    prepare_balanced_hard_dataset,
    HardMiningDataset,
    build_val_loader,
    evaluate_validation,
    build_test_loader,
    predict_probs,
)
# ===================== 单个模型微调 =====================
def finetune_single_model(model_id: int, args, device: torch.device, data_cfg: dict, model_cfg: dict) -> dict:
    """对单个已训练模型: 挖掘困难样本 -> 冻结骨干微调分类头 -> 验证集监控 + 早停 -> 归档。"""
    model_dir = CHECKPOINT_ROOT / f"P{model_id}"
    src_ckpt = model_dir / ORIGINAL_WEIGHT_NAME
    if not src_ckpt.exists():
        print(f"[跳过] P{model_id} 缺少 {ORIGINAL_WEIGHT_NAME}")
        return {}

    run_model_dir = RUN_DIR / f"P{model_id}"
    run_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\n>>> 微调模型 P{model_id} (源: {src_ckpt})\n{'=' * 70}")

    model = DualStreamStripNet(
        pretrained_2d=False,
        feature_dim=int(model_cfg.get("feature_dim", 128)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.5)),
    )
    model = load_model_state(model, str(src_ckpt))
    model.to(device)

    # 1. LOOCV 患者划分
    records = load_training_records()
    train_side, val_side = split_records_by_patient(records, heldout_patient_id=model_id)
    if not val_side:
        print(f"[跳过] P{model_id} 无对应 LOOCV 验证患者 (patient_id={model_id})")
        return {}
    print(f"  训练侧 {len(train_side)} 张 | 验证侧 {len(val_side)} 张")

    # 2. 在训练侧挖掘困难样本
    hard_items, raw_pos, raw_neg = prepare_balanced_hard_dataset(
        model, train_side, device, target_pos=args.target_pos, target_neg=args.target_neg
    )
    print(f"  困难样本: 共 {len(hard_items)} 张 (命中: 阳性 {raw_pos} / 阴性 {raw_neg})")

    # 3. 冻结骨干, 仅微调分类头
    for param in model.image_stream.parameters():
        param.requires_grad = False
    for param in model.proj_stream.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-3)
    pos_weight_tensor = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    hard_ds = HardMiningDataset(hard_items)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(hard_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = build_val_loader(val_side, data_cfg, batch_size=8)

    # 4. 微调前 baseline (验证集) 作为评分起点, 防止微调破坏原始性能
    baseline = evaluate_validation(model, val_loader, criterion, device)
    print(f"  [Baseline] Val Loss={baseline['loss']:.4f} | AUC={baseline['auc']:.4f} | Sens={baseline['sensitivity']:.4f}")

    best_score = baseline["auc"] if not np.isnan(baseline["auc"]) else -baseline["loss"]
    best_epoch = 0
    patience_counter = 0
    best_state = None
    history = []

    # 5. 训练循环 + 早停 (监控验证损失与 AUC)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, projs, labels in loader:
            imgs, projs = imgs.to(device), projs.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs, projs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(loader)

        val_m = evaluate_validation(model, val_loader, criterion, device)
        score = val_m["auc"] if not np.isnan(val_m["auc"]) else -val_m["loss"]
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": val_m["loss"],
            "val_auc": val_m["auc"],
            "val_sensitivity": val_m["sensitivity"],
            "val_specificity": val_m["specificity"],
            "val_accuracy": val_m["accuracy"],
        })
        print(
            f"  Epoch [{epoch:02d}/{args.epochs}] Train {train_loss:.4f} | "
            f"ValLoss {val_m['loss']:.4f} | ValAUC {val_m['auc']:.4f} | ValSens {val_m['sensitivity']:.4f} | "
            f"Patience {patience_counter}/{args.patience}"
        )

        if patience_counter >= args.patience:
            print(f"  ⚠️ 触发早停 (Epoch {epoch})")
            break

    # 6. 归档: 历史数组 + 权重 + 最终验证指标 (配对保存 val_auc / val_sensitivity)
    hist_df = pd.DataFrame(history)
    hist_df.insert(0, "model_id", model_id)
    hist_csv = run_model_dir / "history.csv"
    hist_df.to_csv(hist_csv, index=False)

    save_path = run_model_dir / FINETUNE_WEIGHT_NAME
    if best_state is not None:
        torch.save(best_state, str(save_path))
        print(f"  ✔ 微调最佳 (Epoch {best_epoch}) 权重已保存: {save_path}")
        model.load_state_dict(best_state)
    else:
        # 微调未改善验证指标 -> 保留原始权重, 评估阶段自动回退
        print(f"  [提示] 微调未改善验证指标, 保留原始权重, 不生成微调权重。")
        model = load_model_state(model, str(src_ckpt))

    final_m = evaluate_validation(model, val_loader, criterion, device)
    metrics_row = {
        "model_id": model_id,
        "best_epoch": best_epoch,
        "val_loss": final_m["loss"],
        "val_auc": final_m["auc"],
        "val_sensitivity": final_m["sensitivity"],
        "val_specificity": final_m["specificity"],
        "val_accuracy": final_m["accuracy"],
    }
    pd.DataFrame([metrics_row]).to_csv(run_model_dir / "metrics.csv", index=False)

    print(f"  最终验证: Loss={final_m['loss']:.4f} | AUC={final_m['auc']:.4f} | Sens={final_m['sensitivity']:.4f}")
    print(f"  已归档: {run_model_dir}")
    return metrics_row


def run_finetune_all(model_ids: List[int], args, device: torch.device, data_cfg: dict, model_cfg: dict) -> None:
    """依次微调所有指定模型, 并将每模型最终 val_auc / val_sensitivity 配对汇总保存。"""
    results = []
    for mid in model_ids:
        row = finetune_single_model(mid, args, device, data_cfg, model_cfg)
        if row:
            results.append(row)
    if results:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(results)
        summary_csv = RUN_DIR / "finetune_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print("\n[微调汇总: 最终 val_auc / val_sensitivity 配对]")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"已保存: {summary_csv}")


# ===================== 最终评估: 筛选 + 集成 + 归档 =====================
# 测试集构建/推理/指标计算均已封装至 src/utils/Fine_Tuning_helper.py:
#   build_test_loader / predict_probs / compute_metrics_with_best_threshold


def _save_ensemble_model_checkpoint(
    name_prefix: str, model_ids: List[int], ckpt_used: Dict[int, str], model_cfg: dict
) -> Path:
    """生成并保存一个集成版本的模型参数。

    不依赖测试集 (无需推理), 决策阈值固定 0.5, 避免在测试集上寻优造成泄漏。
    单模型命名保持 P1~P24 序列, 不因筛选重命名。
    """
    ckpt_path = RUN_DIR / f"{name_prefix}_model.pth"
    ensemble_ckpt = {
        "run_name": RUN_NAME,
        "ensemble_type": name_prefix,
        "model_ids": [f"P{i}" for i in model_ids],
        "weights": {f"P{i}": torch.load(ckpt_used[i], map_location="cpu") for i in model_ids},
        "feature_dim": int(model_cfg.get("feature_dim", 128)),
        "dropout_rate": float(model_cfg.get("dropout_rate", 0.5)),
        "ensemble_method": "soft_voting_mean",
        "best_threshold": 0.5,
    }
    torch.save(ensemble_ckpt, str(ckpt_path))
    print(f"  集成[{name_prefix}] 权重已保存: {ckpt_path} (成员 {len(model_ids)} 个)")
    return ckpt_path


def _collect_single_model_info(model_cfg: dict) -> Tuple[List[Dict], Dict[int, str], List[int]]:
    """收集全部已训练模型的权重路径与微调后 LOOCV 验证集指标 (不依赖测试集)。

    模型集合动态扫描 outputs/checkpoints 下实际存在的 P{id}, 有多少就收集多少。
    """
    rows: List[Dict] = []
    ckpt_used: Dict[int, str] = {}
    for model_id in discover_model_ids():
        run_model_dir = RUN_DIR / f"P{model_id}"
        finetuned_ckpt = run_model_dir / FINETUNE_WEIGHT_NAME
        original_ckpt = CHECKPOINT_ROOT / f"P{model_id}" / ORIGINAL_WEIGHT_NAME
        ckpt = finetuned_ckpt if finetuned_ckpt.exists() else original_ckpt
        if not ckpt.exists():
            continue
        ckpt_used[model_id] = str(ckpt)

        # 读取该模型微调后的最终验证指标 (来自 LOOCV held-out 患者, 无泄漏)
        val_auc = val_sens = val_loss = val_acc = best_epoch = np.nan
        metrics_csv = run_model_dir / "metrics.csv"
        if metrics_csv.exists():
            mrow = pd.read_csv(metrics_csv).iloc[0]
            val_auc = float(mrow.get("val_auc", np.nan))
            val_sens = float(mrow.get("val_sensitivity", np.nan))
            val_loss = float(mrow.get("val_loss", np.nan))
            val_acc = float(mrow.get("val_accuracy", np.nan))
            best_epoch = float(mrow.get("best_epoch", np.nan))

        rows.append({
            "model_id": model_id,
            "weight_source": "finetuned" if finetuned_ckpt.exists() else "original",
            "weight_path": str(ckpt),
            "final_val_auc": val_auc,
            "final_val_sensitivity": val_sens,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_acc,
            "best_epoch": best_epoch,
        })
    return rows, ckpt_used, sorted(ckpt_used.keys())


def _save_ensemble_test_outputs(
    name_prefix: str, model_ids: List[int], probs: np.ndarray, labels: np.ndarray, per_model_df: pd.DataFrame
) -> None:
    """保存一个集合版本在测试集上的预测与汇总 (决策阈值固定 0.5, 无泄漏)。"""
    m = compute_metrics(labels, probs, threshold=0.5)
    threshold = 0.5

    pred_df = pd.DataFrame({
        "test_labels": labels,
        "test_probs": probs,
        "pred_label": (probs >= threshold).astype(int),
        "selected_model_ids": [model_ids] * len(labels),
    })
    pred_csv = RUN_DIR / f"{name_prefix}_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    np.save(RUN_DIR / f"{name_prefix}_test_probs.npy", probs)
    np.save(RUN_DIR / f"{name_prefix}_test_labels.npy", labels)

    summary_rows = [
        {
            "type": "ensemble",
            "threshold": threshold,
            "model_ids": str([f"P{i}" for i in model_ids]),
            "auc": m["auc"],
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
        },
    ]
    # 参考: 测试集 AUC 最高的单模型 (仅报告, 不参与筛选/阈值决策, 不构成泄漏)
    valid = per_model_df.dropna(subset=["test_auc"])
    if not valid.empty:
        best = valid.loc[valid["test_auc"].idxmax()]
        summary_rows.append({
            "type": "best_single_ref",
            "threshold": threshold,
            "model_ids": f"P{int(best['model_id'])}",
            "auc": float(best["test_auc"]),
            "accuracy": float(best["test_accuracy"]),
            "sensitivity": float(best["test_sensitivity"]),
            "specificity": float(best["test_specificity"]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RUN_DIR / f"{name_prefix}_summary.csv", index=False)

    print("\n" + "=" * 70)
    print(f">>> 集成[{name_prefix}] (Soft Voting, {len(model_ids)} 个模型) 测试集指标 @阈值 0.5")
    print(f"  AUC={m['auc']:.4f} | Sens={m['sensitivity']:.4f} | Spec={m['specificity']:.4f} | Acc={m['accuracy']:.4f}")
    print(f"  预测已保存: {pred_csv}")
    print(f"  汇总已保存: {RUN_DIR / f'{name_prefix}_summary.csv'}")
    print("=" * 70)


def build_and_save_ensembles(args, device: torch.device, data_cfg: dict, model_cfg: dict) -> Dict:
    """集成构建阶段: 不依赖测试集, 始终执行。

    基于 LOOCV 验证集指标筛选单模型 (无泄漏), 生成筛选版与全量版集合模型权重。
    这样即使 --test-csv 留白, 微调完成后也会完成集成, 不会中途停止。
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\n>>> 集成构建 (基于验证集指标, 不接触测试集)\n{'=' * 70}")

    rows, ckpt_used, all_ids = _collect_single_model_info(model_cfg)
    if not rows:
        print("[警告] 未找到任何可用的单模型权重, 跳过集成构建。")
        return {}

    per_model_df = pd.DataFrame(rows)
    per_model_csv = RUN_DIR / "per_model_validate_metrics.csv"
    per_model_df.to_csv(per_model_csv, index=False)

    print("\n[各单模型验证集指标 (LOOCV held-out, 无泄漏)]")
    print(per_model_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"已保存: {per_model_csv}")

    # 用验证集指标筛选 (而非测试集指标), 避免测试集泄漏
    valid = per_model_df.dropna(subset=["final_val_auc"])
    selected = per_model_df[
        (per_model_df["final_val_auc"].fillna(0.0) > args.ensemble_auc_threshold)
        | (per_model_df["final_val_sensitivity"].fillna(0.0) >= args.ensemble_sens_threshold)
    ]
    selected_ids = selected["model_id"].astype(int).tolist()
    if not selected_ids:
        if not valid.empty:
            best_mid = int(valid.loc[valid["final_val_auc"].idxmax(), "model_id"])
            selected_ids = [best_mid]
            print("[提示] 无模型通过验证集筛选, 退化为验证集 AUC 最高的单模型。")
        else:
            selected_ids = all_ids

    print(f"\n筛选依据: 验证集 final_val_auc>{args.ensemble_auc_threshold} 或 final_val_sensitivity>={args.ensemble_sens_threshold}")
    print(f"筛选版入选 (P 序列): {[f'P{i}' for i in selected_ids]}")
    print(f"全量版: 全部 {len(all_ids)} 个模型")

    # 保存两个集合模型权重 (不依赖测试集)
    _save_ensemble_model_checkpoint("ensemble", selected_ids, ckpt_used, model_cfg)
    _save_ensemble_model_checkpoint("ensemble_all", all_ids, ckpt_used, model_cfg)

    return {
        "ckpt_used": ckpt_used,
        "selected_ids": selected_ids,
        "all_ids": all_ids,
        "per_model_df": per_model_df,
    }


def evaluate_on_test_set(
    args, device: torch.device, data_cfg: dict, model_cfg: dict, build_info: Dict
) -> None:
    """测试集评估阶段: 仅用于最终报告。

    决策阈值固定 0.5 (不在测试集上寻优), 避免泄漏; AUC 无阈值依赖。
    """
    if not build_info:
        print("\n[测试集评估] 集成构建未成功, 跳过。")
        return

    print(f"\n{'=' * 70}\n>>> 测试集评估: {args.test_csv}\n{'=' * 70}")
    test_loader, labels = build_test_loader(args.test_csv, data_cfg, batch_size=8)
    print(f"测试集 {len(labels)} 张 | 阳性 {int(labels.sum())} / 阴性 {int((1 - labels).sum())}")

    ckpt_used = build_info["ckpt_used"]
    selected_ids = build_info["selected_ids"]
    all_ids = build_info["all_ids"]
    val_df = build_info["per_model_df"].set_index("model_id")

    # 单模型测试集指标 (固定 0.5 阈值)
    rows: List[Dict] = []
    all_probs: Dict[int, np.ndarray] = {}
    for model_id in all_ids:
        model = DualStreamStripNet(
            pretrained_2d=False,
            feature_dim=int(model_cfg.get("feature_dim", 128)),
            dropout_rate=float(model_cfg.get("dropout_rate", 0.5)),
        )
        model = load_model_state(model, str(ckpt_used[model_id]))
        model.to(device)
        probs = predict_probs(model, test_loader, device)
        all_probs[model_id] = probs
        m = compute_metrics(labels, probs, threshold=0.5)
        rows.append({
            "model_id": model_id,
            "weight_source": str(val_df.loc[model_id, "weight_source"]),
            "final_val_auc": float(val_df.loc[model_id, "final_val_auc"]),
            "final_val_sensitivity": float(val_df.loc[model_id, "final_val_sensitivity"]),
            "test_auc": m["auc"],
            "test_sensitivity": m["sensitivity"],
            "test_specificity": m["specificity"],
            "test_accuracy": m["accuracy"],
        })

    per_model_df = pd.DataFrame(rows)
    per_model_test_csv = RUN_DIR / "per_model_test_metrics.csv"
    per_model_df.to_csv(per_model_test_csv, index=False)
    print("\n[各单模型测试集指标 @阈值 0.5]")
    print(per_model_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"已保存: {per_model_test_csv}")

    # 两个集合版本在测试集上的评估 (固定 0.5 阈值)
    for name_prefix, mids in [("ensemble", selected_ids), ("ensemble_all", all_ids)]:
        ens_probs = np.mean(np.stack([all_probs[i] for i in mids]), axis=0)
        _save_ensemble_test_outputs(name_prefix, mids, ens_probs, labels, per_model_df)


# ===================== 入口 =====================
def main():
    parser = argparse.ArgumentParser(description="全部已训练模型的困难样本微调与集成评估 (模型数量动态扫描)")
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all",
                        help="执行阶段: train=仅微调, eval=仅最终评估, all=两者 (默认 all)")
    parser.add_argument("--model-ids", type=str, default="all", help="要微调的模型 id, 如 'all' 或 '1,3,5-8'")
    parser.add_argument("--epochs", type=int, default=15, help="微调轮数 (默认 15)")
    parser.add_argument("--lr", type=float, default=1e-4, help="微调学习率 (默认 1e-4)")
    parser.add_argument("--pos-weight", type=float, default=5.0, help="阳性样本损失权重 (默认 5.0)")
    parser.add_argument("--patience", type=int, default=5, help="验证指标无提升时的早停耐心 (默认 5)")
    parser.add_argument("--target-pos", type=int, default=10, help="阳性困难样本目标数量 (默认 10)")
    parser.add_argument("--target-neg", type=int, default=10, help="阴性困难样本目标数量 (默认 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="微调 batch size (默认 4)")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子 (默认 42)")
    parser.add_argument("--test-csv", type=str, default=TEST_CSV,
                        help="独立测试集 CSV 路径 (默认留白, 提供后启用最终评估)")
    parser.add_argument("--ensemble-auc-threshold", type=float, default=0.9,
                        help="集成筛选的测试集 AUC 阈值 (默认 0.9)")
    parser.add_argument("--ensemble-sens-threshold", type=float, default=0.9,
                        help="集成筛选的灵敏度阈值 (默认 0.9)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    model_ids = parse_model_ids(args.model_ids)
    print(f"[*] 设备: {device} | 待微调模型: {[f'P{i}' for i in model_ids]}")
    print(f"[*] 归档目录: {RUN_DIR}")

    if args.stage in ("train", "all"):
        run_finetune_all(model_ids, args, device, data_cfg, model_cfg)

    # 集成构建: 不依赖测试集, 始终执行 (避免 --test-csv 留白时微调完即停止)
    build_info = build_and_save_ensembles(args, device, data_cfg, model_cfg)

    # 测试集评估: 仅当提供测试集时执行 (最终报告)
    if args.stage in ("eval", "all"):
        if args.test_csv and Path(args.test_csv).exists():
            evaluate_on_test_set(args, device, data_cfg, model_cfg, build_info)
        else:
            print("\n[提示] --test-csv 留白: 已完成集成构建(生成集合模型权重), 跳过测试集评估。")


if __name__ == "__main__":
    main()
