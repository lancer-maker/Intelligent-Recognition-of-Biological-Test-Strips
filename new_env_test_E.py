"""
=====================================================================
new_env_test_E.py —— E 组模型测试评估脚本 (配置区驱动, 适配门控融合)
=====================================================================
借鉴 new_env_test_CD.py, 评估逻辑与输出格式完全一致; 区别在于:
  - 模型组固定扫描 outputs/checkpoints/E 下所有模型子目录的 best_model.pth
  - 适配 E 分支模型 = 跨流门控融合双流网络 GatedDualStreamNet
    (src/models/gated_fusion.py): 推理同时使用 图像 + 投影 两个输入,
    与训练时一致 (img -> image_stream; proj -> proj_stream -> GateMLP 门控 img)。
  - 所有输出放到 outputs/checkpoints/E/reports/ 下, 内容与格式同 new_env_test:
      eval_summary.csv            # 全部评估结果
      per_model_eval.csv          # 单模型部分
      ensemble_eval.csv           # Soft Voting 集合部分
      predictions/{name}_preds.csv + {name}_probs.npy   # 逐样本预测 (sample_id 对齐)
  - 决策阈值优先使用训练时基于验证集的阈值 (从各模型 metrics.csv 读取),
    找不到报告才回退测试集约登, 以 threshold_source 列标注来源。
  - 门控网络隐藏层维度 GATE_HIDDEN 从权重本身推断 (gate_mlp.net.2.weight),
    也允许在配置区手动指定兜底值 (应与训练 new_env_train_E.py 配置一致)。

用法: 修改下方配置区后直接运行
  python new_env_test_E.py
=====================================================================
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics
from src.models.gated_fusion import GatedDualStreamNet
from src.utils.model_helper import load_model_state_dict
from src.utils.Fine_Tuning_helper import (
    build_test_loader,
    compute_metrics_with_best_threshold,
    load_config,
)
from new_env_test import (        # 复用 new_env_test 的评估逻辑/函数
    _metric_row,
    build_voting_ensemble,
    load_validation_threshold,
    make_name,
    natural_sort_key,
    save_predictions,
)

# ================= 配置区 =================
CHECKPOINT_NAME = "E"                       # 模型组 (对应 outputs/checkpoints/E)
TEST_CSV = r"data\test\labels\labels.csv"   # 独立测试集 CSV 路径
TEST_RAW_DIR = ""                           # 测试集 raw 目录 (可选; 默认自动探测 CSV 同级/上一级 raw)
BATCH_SIZE = 8                              # 推理 batch size
DO_ENSEMBLE_VOTE = True                     # 是否对多个单模型做 Soft Voting 集合评估
GATE_HIDDEN_FALLBACK = 64                   # 兜底门控隐藏维度 (优先从权重推断)
# ==========================================

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / CHECKPOINT_NAME
REPORTS_DIR = CHECKPOINT_DIR / "reports"    # 输出目录: outputs/checkpoints/E/reports


def discover_models(checkpoint_dir: Path):
    """扫描 checkpoint_dir 下所有 {id}/best_model.pth (按目录名自然排序)。"""
    return sorted(checkpoint_dir.glob("*/best_model.pth"),
                  key=lambda p: natural_sort_key(p.parent.name))


def _infer_gate_hidden(state_dict: dict, fallback: int = GATE_HIDDEN_FALLBACK) -> int:
    """从权重推断门控网络 GateMLP 隐藏层维度。

    GateMLP = Linear(fd, hidden) -> ReLU -> Linear(hidden, fd) -> Sigmoid,
    故 gate_mlp.net.2.weight 形状为 (fd, hidden), 隐藏维 = shape[1]。
    权重缺失/异常时回退 fallback。
    """
    w = state_dict.get("gate_mlp.net.2.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1])
    return int(fallback)


def evaluate_gated_weight(weight_path, loader, labels, device, model_cfg):
    """评估 E 分支的跨流门控融合模型 (GatedDualStreamNet, 需 图像+投影)。

    决策阈值优先用训练时验证集阈值 (同目录 metrics.csv), 输出格式与
    CD 版 (evaluate_image_only_weight / evaluate_single_weight) 完全一致。
    """
    feature_dim = int(model_cfg.get("feature_dim", 128))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.5))
    state_dict = load_model_state_dict(str(weight_path))
    gate_hidden = _infer_gate_hidden(state_dict)
    model = GatedDualStreamNet(
        pretrained_2d=False,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate,
        gate_hidden=gate_hidden,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    probs_list = []
    with torch.no_grad():
        for imgs, projs, _ in loader:
            imgs, projs = imgs.to(device), projs.to(device)
            logits = model(imgs, projs)                          # (B, 1) logits
            probs_list.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
    probs = np.asarray(probs_list, dtype=np.float32)

    m = compute_metrics_with_best_threshold(labels, probs)       # auc + @0.5 + 测试集约登参考
    val = load_validation_threshold(weight_path)
    if val is not None:
        th, lo, hi = val
        source = "validation_bootstrap_youden"
        m_th = compute_metrics(labels, probs, threshold=th)
        sens, spec, acc = m_th["sensitivity"], m_th["specificity"], m_th["accuracy"]
        print(f"    [阈值] 使用训练时验证集阈值 {th:.4f} (来源: {weight_path.parent / 'metrics.csv'})")
    else:
        th, lo, hi = m["best_threshold"], np.nan, np.nan
        source = "test_youden"
        sens, spec, acc = m["sensitivity"], m["specificity"], m["accuracy"]
        print(f"    [警告] 未找到对应验证集报告 {weight_path.parent / 'metrics.csv'}, "
              f"回退测试集约登阈值 {th:.4f}")

    row = _metric_row(make_name(weight_path), weight_path, m["auc"], th, source,
                      sens, spec, acc, m["sens_05"], m["spec_05"], m["acc_05"],
                      n_models=1, model_ids=str([weight_path.name]),
                      th_ci_low=lo, th_ci_high=hi)
    return row, probs


def main():
    # 1. 校验
    if not CHECKPOINT_DIR.exists():
        print(f"[错误] 模型目录不存在: {CHECKPOINT_DIR}")
        return
    if not Path(TEST_CSV).exists() and not Path(str(TEST_CSV) + ".csv").exists():
        print(f"[错误] 测试集 CSV 不存在: {TEST_CSV}")
        return

    # 2. 自动收集模型 (目录里有多少就评估多少)
    model_paths = discover_models(CHECKPOINT_DIR)
    if not model_paths:
        print(f"[错误] {CHECKPOINT_DIR} 下未找到任何 best_model.pth")
        return
    print(f"[*] 模型组: {CHECKPOINT_NAME} (门控融合 GatedDualStreamNet) | 发现模型: {len(model_paths)} 个")
    for p in model_paths:
        print(f"    - {p}")

    # 3. 设备/配置/测试集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    test_loader, labels = build_test_loader(
        TEST_CSV, data_cfg, batch_size=BATCH_SIZE, image_dir=TEST_RAW_DIR
    )
    print(f"[*] 设备: {device} | 测试集: {TEST_CSV} | 样本数 {len(labels)} "
          f"(阳性 {int(labels.sum())} / 阴性 {int((1 - labels).sum())})")
    print(f"[*] 输出目录: {REPORTS_DIR}")

    # 4. 逐模型评估 (阈值优先用训练时验证集阈值)
    rows = []
    single_entries = []
    for idx, wp in enumerate(model_paths, start=1):
        print(f"\n[{idx}/{len(model_paths)}] 评估: {wp}")
        ret = evaluate_gated_weight(wp, test_loader, labels, device, model_cfg)
        if ret is None:
            continue
        row, probs = ret
        rows.append(row)
        save_predictions(REPORTS_DIR, row["name"], labels, probs, row["best_threshold"])
        single_entries.append((row["name"], wp, probs))
        print(f"    AUC={row['auc']:.4f} | thr={row['best_threshold']:.4f} "
              f"[{row['threshold_source']}] | Sens={row['sensitivity']:.4f} Spec={row['specificity']:.4f}")

    if not rows:
        print("[错误] 所有模型均评估失败, 退出。")
        return

    # 5. Soft Voting 集合评估 (保留集合模型测试方式)
    if DO_ENSEMBLE_VOTE and len(single_entries) >= 2:
        print(f"\n[集合评估] 对 {len(single_entries)} 个单模型做 Soft Voting 概率平均")
        ens_row, ens_probs = build_voting_ensemble("ensemble_vote", single_entries, labels)
        rows.append(ens_row)
        save_predictions(REPORTS_DIR, ens_row["name"], labels, ens_probs, ens_row["best_threshold"])
        print(f"    AUC={ens_row['auc']:.4f} | thr={ens_row['best_threshold']:.4f} "
              f"[{ens_row['threshold_source']}] | Sens={ens_row['sensitivity']:.4f} Spec={ens_row['specificity']:.4f}")

    # 6. 写报告 (内容与格式同 new_env_test, 全部放进 reports 目录)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(REPORTS_DIR / "eval_summary.csv", index=False)

    per_model = summary_df[summary_df["n_models"] == 1]
    if not per_model.empty:
        per_model.to_csv(REPORTS_DIR / "per_model_eval.csv", index=False)
    ensemble = summary_df[summary_df["n_models"] > 1]
    if not ensemble.empty:
        ensemble.to_csv(REPORTS_DIR / "ensemble_eval.csv", index=False)

    print("\n" + "=" * 70)
    print(f">>> {CHECKPOINT_NAME} 组模型测试评估完成 | 输出目录: {REPORTS_DIR}")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 70)


if __name__ == "__main__":
    main()
