"""
final_test.py —— 集合模型与单模型最终评估脚本
==============================================
作用:
  1. 扫描 outputs/reports/test/ 下所有 ensemble_finetune* 运行目录 (按序号自然排序),
     评估每个目录中的集合模型权重 (ensemble_model.pth 筛选版 / ensemble_all_model.pth 全量版 等)。
  2. 按测试集指标选出最好的集合模型。
  3. 以最好的集合模型所在运行目录为基准, 检测全部单独的模型:
       权重优先取该运行目录下 P{id}/finetuned_classifier.pth (微调后参数),
       若不存在则回退到 outputs/checkpoints/P{id}/best_model.pth。
  4. 所有评估结果输出到 outputs/reports/test/final_test/。

用法示例:
  python final_test.py --test-csv "data/test/labels/labels.csv"
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state
from src.utils.Fine_Tuning_helper import (
    CHECKPOINT_ROOT,
    TEST_REPORT_ROOT,
    build_test_loader,
    compute_metrics_with_best_threshold,
    discover_model_ids,
    load_config,
    predict_probs,
)

# ===================== 常量配置 =====================
FINAL_TEST_DIR = TEST_REPORT_ROOT / "final_test"   # 最终结果输出目录
RUN_PREFIX = "ensemble_finetune"                   # 集合模型运行目录前缀
FINETUNE_WEIGHT_NAME = "finetuned_classifier.pth"  # 微调后权重文件名
ORIGINAL_WEIGHT_NAME = "best_model.pth"            # 原始训练权重文件名
DEFAULT_TEST_CSV = "data/test/labels/labels.csv"   # 默认独立测试集 CSV


def natural_sort_key(name: str) -> List:
    """自然排序键: 使 ensemble_finetune2 排在 ensemble_finetune10 之前。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def discover_run_dirs() -> List[Path]:
    """扫描 outputs/reports/test 下所有 ensemble_finetune* 目录并按序号自然排序。"""
    dirs = [d for d in TEST_REPORT_ROOT.glob(f"{RUN_PREFIX}*") if d.is_dir()]
    return sorted(dirs, key=lambda d: natural_sort_key(d.name))


def discover_ensemble_weights(run_dir: Path) -> List[Path]:
    """查找运行目录顶层的集合模型权重 (ensemble*_model.pth, 不递归 P 子目录)。"""
    return sorted(run_dir.glob("ensemble*_model.pth"))


def extract_state_dict(ckpt: dict) -> dict:
    """从 checkpoint 内容中提取裸 state_dict (兼容裸 state_dict 与 {'model_state_dict': ...})。"""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def build_model(feature_dim: int, dropout_rate: float, device: torch.device) -> DualStreamStripNet:
    """按给定特征维度/丢弃率构建双流网络并移至设备。"""
    model = DualStreamStripNet(pretrained_2d=False, feature_dim=feature_dim, dropout_rate=dropout_rate)
    return model.to(device)


def evaluate_ensemble_weight(
    weight_path: Path,
    run_name: str,
    loader,
    labels: np.ndarray,
    device: torch.device,
) -> Optional[Tuple[Dict, np.ndarray]]:
    """评估一个集合模型权重: 加载其包含的成员权重, 逐成员推理后概率平均 (Soft Voting)。

    Returns:
        (指标行 dict, 集合预测概率 np.ndarray); 无有效成员时返回 None
    """
    ckpt = torch.load(weight_path, map_location="cpu")
    feature_dim = int(ckpt.get("feature_dim", 128))
    dropout_rate = float(ckpt.get("dropout_rate", 0.5))
    model_ids = ckpt.get("model_ids") or []
    weights = ckpt.get("weights") or {}

    probs_list: List[np.ndarray] = []
    used_ids: List = []
    for pid in model_ids:
        state = weights.get(pid)
        if state is None:
            continue
        model = build_model(feature_dim, dropout_rate, device)
        model.load_state_dict(extract_state_dict(state))
        probs_list.append(predict_probs(model, loader, device))
        used_ids.append(pid)

    if not probs_list:
        print(f"[警告] 集合权重 {weight_path} 无有效成员权重, 跳过。")
        return None

    probs = np.mean(np.stack(probs_list), axis=0)
    m = compute_metrics_with_best_threshold(labels, probs)

    row = {
        "run_name": run_name,
        "weight_name": weight_path.name,
        "n_models": len(used_ids),
        "model_ids": str(used_ids),
        "auc": m["auc"],
        "best_threshold": m["best_threshold"],
        "accuracy": m["accuracy"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "sens_05": m["sens_05"],
        "spec_05": m["spec_05"],
        "acc_05": m["acc_05"],
        "weight_path": str(weight_path),
    }
    return row, probs


def evaluate_single_models(
    best_run_dir: Path,
    loader,
    labels: np.ndarray,
    device: torch.device,
    model_cfg: dict,
) -> List[Dict]:
    """评估所有单独模型: 优先使用 best_run_dir 下的微调权重, 缺失则回退 checkpoints 原始权重。

    模型集合动态扫描实际存在的 P{id}, 有多少就评估多少。
    """
    rows: List[Dict] = []
    feature_dim = int(model_cfg.get("feature_dim", 128))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.5))

    for model_id in discover_model_ids():
        finetuned = best_run_dir / f"P{model_id}" / FINETUNE_WEIGHT_NAME
        original = CHECKPOINT_ROOT / f"P{model_id}" / ORIGINAL_WEIGHT_NAME

        if finetuned.exists():
            ckpt, src = finetuned, "finetuned"
        elif original.exists():
            ckpt, src = original, "original"
        else:
            print(f"[跳过] P{model_id} 无任何可用权重。")
            continue

        model = DualStreamStripNet(
            pretrained_2d=False, feature_dim=feature_dim, dropout_rate=dropout_rate
        )
        model = load_model_state(model, str(ckpt))
        model.to(device)
        probs = predict_probs(model, loader, device)
        m = compute_metrics_with_best_threshold(labels, probs)

        rows.append({
            "model_id": model_id,
            "weight_source": src,
            "weight_path": str(ckpt),
            "auc": m["auc"],
            "best_threshold": m["best_threshold"],
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "sens_05": m["sens_05"],
            "spec_05": m["spec_05"],
            "acc_05": m["acc_05"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="集合模型与单模型最终评估")
    parser.add_argument("--test-csv", type=str, default=DEFAULT_TEST_CSV,
                        help=f"独立测试集 CSV 路径 (默认 {DEFAULT_TEST_CSV})")
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch size (默认 8)")
    args = parser.parse_args()

    if not Path(args.test_csv).exists():
        print(f"[错误] 测试集 CSV 不存在: {args.test_csv}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    test_loader, labels = build_test_loader(args.test_csv, data_cfg, batch_size=args.batch_size)
    print(f"[*] 设备: {device} | 测试集: {args.test_csv} | 样本数 {len(labels)} "
          f"(阳性 {int(labels.sum())} / 阴性 {int((1 - labels).sum())})")
    print(f"[*] 输出目录: {FINAL_TEST_DIR}")

    # 1. 扫描并评估所有运行目录中的集合模型
    run_dirs = discover_run_dirs()
    print(f"\n[1/3] 扫描到 {len(run_dirs)} 个运行目录: {[d.name for d in run_dirs]}")

    ens_rows: List[Dict] = []
    probs_by_key: Dict[str, np.ndarray] = {}
    for run_dir in run_dirs:
        for w in discover_ensemble_weights(run_dir):
            print(f"  评估集合模型: {run_dir.name}/{w.name}")
            ret = evaluate_ensemble_weight(w, run_dir.name, test_loader, labels, device)
            if ret is None:
                continue
            row, probs = ret
            ens_rows.append(row)
            probs_by_key[f"{run_dir.name}/{w.name}"] = probs

    if not ens_rows:
        print("[错误] 未找到任何集合模型权重, 退出。")
        return

    FINAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    ens_df = pd.DataFrame(ens_rows)
    ens_csv = FINAL_TEST_DIR / "ensemble_models_eval.csv"
    ens_df.to_csv(ens_csv, index=False)
    print(f"\n[各集合模型测试集指标]")
    print(ens_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"已保存: {ens_csv}")

    # 2. 选出最好的集合模型 (按测试集 AUC 最大)
    valid_df = ens_df.dropna(subset=["auc"])
    if valid_df.empty:
        print("[错误] 所有集合模型 AUC 均不可用 (测试集可能单类), 退出。")
        return
    best = valid_df.loc[valid_df["auc"].idxmax()]
    best_key = f"{best['run_name']}/{best['weight_name']}"
    best_probs = probs_by_key.get(best_key)
    best_run_dir = TEST_REPORT_ROOT / str(best["run_name"])
    print(f"\n[2/3] 最好的集合模型: {best_key}  (AUC={best['auc']:.4f}, Sens={best['sensitivity']:.4f})")

    best_summary = pd.DataFrame([{
        "best_run_name": best["run_name"],
        "best_weight_name": best["weight_name"],
        "n_models": best["n_models"],
        "model_ids": best["model_ids"],
        "auc": best["auc"],
        "best_threshold": best["best_threshold"],
        "accuracy": best["accuracy"],
        "sensitivity": best["sensitivity"],
        "specificity": best["specificity"],
        "sens_05": best["sens_05"],
        "spec_05": best["spec_05"],
        "acc_05": best["acc_05"],
    }])
    best_summary.to_csv(FINAL_TEST_DIR / "best_ensemble_summary.csv", index=False)

    # 保存最好集合模型的预测 (test_probs / test_labels)
    if best_probs is not None:
        pred_df = pd.DataFrame({
            "test_labels": labels,
            "test_probs": best_probs,
            "pred_label": (best_probs >= float(best["best_threshold"])).astype(int),
            "selected_model_ids": [str(best["model_ids"])] * len(labels),
        })
        pred_df.to_csv(FINAL_TEST_DIR / "best_ensemble_predictions.csv", index=False)
        np.save(FINAL_TEST_DIR / "best_ensemble_test_probs.npy", best_probs)
        np.save(FINAL_TEST_DIR / "best_ensemble_test_labels.npy", labels)

    # 3. 以最好集合模型所在运行目录为基准, 检测全部单独模型
    single_ids = discover_model_ids()
    print(f"\n[3/3] 检测 {len(single_ids)} 个单独模型 (权重优先 {best_run_dir}/P{{id}}/{FINETUNE_WEIGHT_NAME}, 缺失回退 checkpoints)")
    per_model_rows = evaluate_single_models(best_run_dir, test_loader, labels, device, model_cfg)
    per_model_df = pd.DataFrame(per_model_rows)
    per_model_csv = FINAL_TEST_DIR / "per_model_eval.csv"
    per_model_df.to_csv(per_model_csv, index=False)
    print(f"\n[{len(per_model_df)} 个单独模型测试集指标]")
    print(per_model_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"已保存: {per_model_csv}")

    # 最终汇总打印
    print("\n" + "=" * 70)
    print(">>> 最终评估结果汇总")
    print(f"  最好集合模型: {best_key}")
    print(f"    AUC={best['auc']:.4f} | Sens={best['sensitivity']:.4f} | Spec={best['specificity']:.4f} | Acc={best['accuracy']:.4f}")
    if not per_model_df.empty:
        pm_valid = per_model_df.dropna(subset=["auc"])
        if not pm_valid.empty:
            pm_best = pm_valid.loc[pm_valid["auc"].idxmax()]
            print(f"  最好单模型: P{int(pm_best['model_id'])} (来源 {pm_best['weight_source']}) | "
                  f"AUC={pm_best['auc']:.4f} | Sens={pm_best['sensitivity']:.4f}")
        print(f"  单模型 AUC 均值: {per_model_df['auc'].mean():.4f} ± {per_model_df['auc'].std():.4f}")
    print(f"  输出目录: {FINAL_TEST_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
