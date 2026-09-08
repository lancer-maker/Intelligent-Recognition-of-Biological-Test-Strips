"""
=====================================================================
figure_result.py —— 论文常用"三联图"绘制模块
    (Loss 曲线 | 混淆矩阵 | PR 曲线)
=====================================================================
设计要点 (符合论文三联图排版要求):
  - 画布 1x3, figsize=(18, 6), 保存 dpi=300
  - 全局字体 serif (Times New Roman / STIXGeneral), 字号 14~16
  - 图1(左) Loss 曲线: 原始 results(蓝实线+o, 置顶) + smooth(虚线粗线)
  - 图2(中) 混淆矩阵: sns.heatmap(annot, fmt='.2f', cmap='Blues', cbar)
              X 轴=Predicted, Y 轴=True (标准布局); 行归一化 => 对角线=各类召回率
  - 图3(右) PR 曲线: 各类别蓝色系曲线 + 全类别平均线(mAP)

数据输入格式 (供 plot_three_panel 使用):
  x_axis     : 训练步数数组 (如 0~400)
  results    : 原始训练 Loss 数组
  smooth     : 平滑后的训练 Loss 数组
  val_loss   : 验证 Loss 数组 (可选; 提供则额外绘制验证损失曲线)
  conf_matrix: N×N 混淆矩阵 (行=True, 列=Predicted, 行归一化, 对角线=各类召回率)
               classes: N 个类别名列表
  pr_data    : {类别: (recall 数组, precision 数组)}
  # 类别 AP / mAP 在绘图时统一在 [0,1] recall 网格上插值计算,
  # 平均 PR 曲线下面积即为 mAP (与图例/标注数值一致)

提供两种数据来源:
  1) make_mock_inputs()  : 生成符合上述格式的模拟数据 (便于直接运行/联调)
  2) load_real_inputs()  : 从本项目真实 CSV 输出读取并"新增数据输出"
     (本项目原输出缺 混淆矩阵/逐类 PR/mAP, 因此该函数会推导并写出:
      loss_curve.csv / confusion_matrix.csv / pr_curves.csv / pr_metrics.csv)

用法:
  # 纯模拟数据演示
  python -m src.utils.figure_result --mode mock
  # 真实数据 (默认模型 5, 读取 Tweak/{id}/history.csv + new_env_test predictions)
  python -m src.utils.figure_result --mode real --model-id 5
=====================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import (
        confusion_matrix as sk_confusion_matrix,
        precision_recall_curve,
    )
except Exception:  # pragma: no cover
    precision_recall_curve = None
    sk_confusion_matrix = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------- 全局样式 ---------------------
PLOT_FIGSIZE = (18, 6)
PLOT_DPI = 300
BLUES = ["#1f77b4", "#aec7e8", "#3182bd", "#6baed6", "#7aa6d6",
         "#4f81bd", "#9ecae1", "#c6dbef", "#2f6f9f", "#d0e1f9"]


def apply_global_style(font_size: int = 14, label_size: int = 16) -> None:
    """设置论文风格全局字体与字号 (serif / Times New Roman / STIXGeneral)。"""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["axes.titlesize"] = label_size + 1
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = PLOT_DPI
    plt.rcParams["axes.grid"] = False


def _moving_average(values: np.ndarray, window: int = 15) -> np.ndarray:
    """简单滑动平均平滑 (两端边界用可用窗口)。"""
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2 - (window % 2 == 0)), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    # 对齐长度 (卷积 valid 可能少几个点, 边界裁剪)
    if len(smoothed) > len(values):
        start = (len(smoothed) - len(values)) // 2
        smoothed = smoothed[start:start + len(values)]
    return smoothed[:len(values)]


# --------------------- 核心绘图 ---------------------
def plot_three_panel(
    x_axis: Sequence[float],
    results: Sequence[float],
    smooth: Sequence[float],
    conf_matrix: np.ndarray,
    classes: List[str],
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_loss: Optional[Sequence[float]] = None,
    save_path: Optional[Path] = None,
    titles: Optional[List[str]] = None,
) -> plt.Figure:
    """绘制三联图 (Loss | Confusion | PR), 返回 Figure 对象。

    Args:
        x_axis     : 训练步数数组
        results    : 原始训练 Loss
        smooth     : 平滑后的训练 Loss
        val_loss   : 验证 Loss (可选; 非空则额外绘制验证损失曲线)
        conf_matrix: N×N 混淆矩阵 (行=True, 列=Predicted, 行归一化 -> 对角线=各类召回率)
        classes    : 类别名列表
        pr_data    : {类别: (recall, precision)}
        save_path  : 非空则保存 PNG (dpi=300)
        titles     : 三个子图标题 (默认 "Loss Curve"/"Confusion Matrix"/"PR Curves")
    """
    apply_global_style()

    titles = titles or ["Loss Curve", "Confusion Matrix", "PR Curves"]
    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE)

    x_axis = np.asarray(x_axis, dtype=float)
    results = np.asarray(results, dtype=float)
    smooth = np.asarray(smooth, dtype=float)

    # ---------- 图1 (左) Loss 曲线 (训练/平滑/验证) ----------
    ax = axes[0]
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.plot(x_axis, results, color="#1f77b4", marker="o", markersize=3,
            linestyle="-", linewidth=1.5, label="Train Loss (raw)", zorder=3)
    ax.plot(x_axis, smooth, color="#ffa64d", linestyle="--", linewidth=2,
            label="Train Loss (smooth)", zorder=2)
    if val_loss is not None:
        val_arr = np.asarray(val_loss, dtype=float)
        if len(val_arr) == len(x_axis):
            ax.plot(x_axis, val_arr, color="#2ca02c", linestyle=":",
                    linewidth=2, label="Val Loss", zorder=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    ax.set_title(titles[0])

    # ---------- 图2 (中) 混淆矩阵 (行=True, 列=Predicted; 行归一化 -> 对角线=召回率) ----------
    ax = axes[1]
    conf = np.asarray(conf_matrix, dtype=float)
    sns.heatmap(conf, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(titles[1])

    # ---------- 图3 (右) PR 曲线 (统一网格插值 + 平均线, 面积 = mAP) ----------
    ax = axes[2]
    grid, mean_prec, ap_cls, mAP = _pr_grid_summary(classes, pr_data, n=101)
    for i, cls in enumerate(classes):
        if cls not in pr_data:
            continue
        rec, pre = pr_data[cls]
        color = BLUES[i % len(BLUES)]
        label = f"{cls} {ap_cls[cls]:.3f}" if cls in ap_cls else cls
        ax.plot(rec, pre, color=color, linewidth=1.5, label=label)
    if len(ap_cls) > 0:
        ax.plot(grid, mean_prec, color="#6a0dad", linewidth=2.2,
                linestyle="-", label=f"all classes {mAP:.3f} mAP@0.5", zorder=4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="best")
    ax.set_title(titles[2])

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    return fig


def _interpolate_precision(recall: np.ndarray, precision: np.ndarray,
                           grid: np.ndarray) -> np.ndarray:
    """标准 PR 插值: 在给定 recall 网格 g 上取"右侧最大 precision" (单调包络)。

    sklearn 的 precision_recall_curve 返回的 precision 随阈值非单调;
    标准绘图/求 AP 的做法是取其单调递减包络: P(g) = max{ precision_i | recall_i >= g }。
    """
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    grid = np.asarray(grid, dtype=float)
    out = np.empty_like(grid)
    for j, g in enumerate(grid):
        mask = recall >= g
        out[j] = float(precision[mask].max()) if mask.any() else 1.0
    return out


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """数值梯形积分 (兼容 numpy>=2 的 np.trapezoid)。"""
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(y, x))


def _pr_grid_summary(classes: List[str],
                     pr_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                     n: int = 101) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
    """标准平均 PR 曲线与 mAP (全部在同一 recall 网格上插值)。

    对每个类别在 [0,1] 均匀网格 (101 点) 上取插值 precision (右侧最大),
    再对所有类别在同一 recall 点的 precision 求平均得到平均 PR 曲线;
    其梯形下面积即 mAP, 且 == 各类别插值 AP 的均值 (与图例数值完全一致)。
    """
    grid = np.linspace(0.0, 1.0, n)
    interp: Dict[str, np.ndarray] = {}
    for cls in classes:
        if cls not in pr_data:
            continue
        rec, pre = pr_data[cls]
        interp[cls] = _interpolate_precision(rec, pre, grid)
    if not interp:
        return grid, np.zeros(n), {}, 0.0
    ap_cls = {cls: _trapz(interp[cls], grid) for cls in interp}
    mean_prec = np.mean(np.stack(list(interp.values())), axis=0)
    mAP = _trapz(mean_prec, grid)     # == mean(ap_cls)
    return grid, mean_prec, ap_cls, mAP


# --------------------- 模拟数据 ---------------------
def make_mock_inputs() -> dict:
    """生成符合格式要求的模拟数据, 便于模块开箱即用/联调。"""
    rng = np.random.default_rng(42)

    # 1) Loss: 训练步数 0~400, 指数下降 + 噪声, 再平滑; 另造一条验证损失(略高于训练)
    x_axis = np.linspace(0.0, 400.0, 401)
    base = 0.75 * np.exp(-x_axis / 130.0) + 0.06
    results = base + rng.normal(0.0, 0.02, size=len(x_axis))
    smooth = _moving_average(results, window=15)
    val_loss = base + 0.10 + rng.normal(0.0, 0.02, size=len(x_axis))

    # 2) 混淆矩阵: 5 类, 行=True, 列=Predicted; 行归一化 -> 对角线=各类召回率
    classes = ["N", "P", "Qc", "Qw", "Background"]
    counts = np.array([
        [95, 2, 1, 0, 2],
        [1, 88, 2, 1, 0],
        [1, 2, 90, 3, 1],
        [1, 1, 3, 92, 2],
        [2, 1, 1, 2, 93],
    ], dtype=float)
    conf_matrix = counts / counts.sum(axis=1, keepdims=True)

    # 3) PR 数据: 每类 300 样本, 得分略含噪声 + 每类不同可分性 -> 曲线有区分且 AP≈0.9x
    pr_data, map_values = {}, {}
    n_samp = 300
    y_global = np.zeros((n_samp, len(classes)), dtype=int)
    for c in range(len(classes)):
        y_global[rng.choice(n_samp, size=120, replace=False), c] = 1
    for c, cls in enumerate(classes):
        y = y_global[:, c]
        margin = float(rng.uniform(0.55, 0.95))       # 每类可分性不同
        score = y.astype(float) * margin + rng.normal(0.0, 0.22, size=n_samp)
        score = np.clip(score, 0.0, 1.0)
        prec, rec, _ = precision_recall_curve(y, score)
        pr_data[cls] = (rec, prec)
    # AP / mAP 由统一 recall 网格插值计算 (与平均 PR 曲线下面积一致)
    _, _, map_values, all_map = _pr_grid_summary(classes, pr_data, n=101)

    return {
        "x_axis": x_axis, "results": results, "smooth": smooth, "val_loss": val_loss,
        "conf_matrix": conf_matrix, "classes": classes,
        "pr_data": pr_data, "map_values": map_values, "all_map": all_map,
    }


# --------------------- 真实数据 (新增数据输出) ---------------------
def _moving_average_windowed(values: np.ndarray, window: int = 9) -> np.ndarray:
    """针对 1D 数组的滑窗平均 (与通用平滑一致, 便于写回 CSV)。"""
    return _moving_average(values, window=window)


def load_real_inputs(model_id: int = 5) -> dict:
    """从本项目真实输出读取并推导三联图输入, 同时把缺失数据写成新 CSV。

    数据来源:
      - Loss 曲线: outputs/checkpoints/Tweak/{id}/history.csv  (train_loss 逐 epoch)
      - 混淆矩阵/PR: outputs/reports/new_env_test/predictions/{id}_best_model_preds.csv
        (该预测 CSV 仅含逐样本 label/prob/pred, 混淆矩阵与逐类 PR/AP/mAP 由此推导)

    新增数据输出目录:
      outputs/reports/three_panel/model{id}/loss_curve.csv / confusion_matrix.csv
      / pr_metrics.csv / pr_curves.csv
    """
    history_csv = PROJECT_ROOT / "outputs" / "checkpoints" / "Tweak" / str(model_id) / "history.csv"
    pred_csv = (PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "predictions"
                / f"{model_id}_best_model_preds.csv")
    out_dir = PROJECT_ROOT / "outputs" / "reports" / "three_panel" / f"model{model_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not history_csv.exists():
        raise FileNotFoundError(f"找不到训练 history: {history_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"找不到测试预测: {pred_csv}")

    # ---------- 1) Loss 曲线 (逐 epoch 累计, 含 Stage1/2) ----------
    hist = pd.read_csv(history_csv)
    train_loss = hist["train_loss"].astype(float).to_numpy()
    x_axis = np.arange(len(train_loss), dtype=float)      # 累计步/epoch 序号(0 起)
    smooth = _moving_average_windowed(train_loss, window=9)
    if "val_loss" in hist.columns:
        # 验证损失可能只在部分 epoch 有值 (val_interval>1), 插值填充使曲线连续
        val_loss = (pd.to_numeric(hist["val_loss"], errors="coerce")
                    .interpolate(limit_direction="both").to_numpy())
    else:
        val_loss = np.full(len(train_loss), np.nan)
    loss_df = pd.DataFrame({"step": x_axis.astype(int),
                            "stage": hist["stage"].to_numpy(),
                            "epoch": hist["epoch"].to_numpy(),
                            "results": train_loss, "smooth": smooth,
                            "val_loss": val_loss})
    loss_df.to_csv(out_dir / "loss_curve.csv", index=False, encoding="utf-8-sig")

    # ---------- 2) 测试预测 -> 混淆矩阵 + 逐类 PR ----------
    pred = pd.read_csv(pred_csv)
    y_true = pred["test_labels"].astype(int).to_numpy()
    y_prob = pred["test_probs"].astype(float).to_numpy()
    y_pred = pred["pred_label"].astype(int).to_numpy()

    # 本项目为二分类: 0=N(阴性), 1=P(阳性)
    classes = ["N", "P"]
    # sklearn confusion_matrix: 行=True, 列=Predicted; 行归一化 -> 对角线=各类召回率
    cm_raw = sk_confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    conf_matrix = cm_raw / row_sums if row_sums.sum() > 0 else cm_raw

    cm_out = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    cm_out.index.name = "True"
    cm_out.columns.name = "Predicted"
    cm_out.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    # PR / AP: 类别 P(1) 用原始概率; 类别 N(0) 用 1-概率 (one-vs-rest)
    pr_data, map_values = {}, {}
    pr_rows = []
    for i, cls in enumerate(classes):
        if cls == "P":
            y_bin = (y_true == 1).astype(int)
            score = y_prob
        else:  # N
            y_bin = (y_true == 0).astype(int)
            score = 1.0 - y_prob
        prec, rec, _ = precision_recall_curve(y_bin, score)
        pr_data[cls] = (rec, prec)
        for r, p in zip(rec, prec):
            pr_rows.append({"class": cls, "recall": float(r), "precision": float(p)})
    # AP / mAP 由统一 recall 网格插值计算 (与平均 PR 曲线下面积一致)
    _, _, map_values, all_map = _pr_grid_summary(classes, pr_data, n=101)

    pd.DataFrame(pr_rows).to_csv(out_dir / "pr_curves.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"class": cls, "AP": map_values[cls]} for cls in classes]
                 + [{"class": "all", "AP": all_map}]
                 ).to_csv(out_dir / "pr_metrics.csv", index=False, encoding="utf-8-sig")

    return {
        "x_axis": x_axis, "results": train_loss, "smooth": smooth, "val_loss": val_loss,
        "conf_matrix": conf_matrix, "classes": classes,
        "pr_data": pr_data, "map_values": map_values, "all_map": all_map,
        "out_dir": out_dir,
    }


# --------------------- 供外部调用 ---------------------
def export_real_figure(model_id: int = 5, out_png: Optional[str] = None) -> str:
    """生成某模型 (真实数据) 的三联图并"新增数据输出", 返回 PNG 路径。

    供 new_env_test.py 等测试/评估脚本调用 (评估完成后自动出图)。
    """
    data = load_real_inputs(model_id=int(model_id))
    png = Path(out_png) if out_png else data["out_dir"] / f"three_panel_model{int(model_id)}.png"
    fig = plot_three_panel(
        x_axis=data["x_axis"], results=data["results"], smooth=data["smooth"],
        val_loss=data.get("val_loss"), conf_matrix=data["conf_matrix"],
        classes=data["classes"], pr_data=data["pr_data"], save_path=png,
    )
    plt.close(fig)
    return str(png)


# --------------------- 入口 ---------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="论文三联图 (Loss | Confusion | PR)")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock",
                        help="mock=模拟数据(默认); real=读取本项目真实 CSV 并新增输出")
    parser.add_argument("--model-id", type=int, default=5, help="real 模式使用的模型编号 (默认 5)")
    parser.add_argument("--out", type=str, default="",
                        help="输出 PNG 路径 (默认 outputs/reports/three_panel/)")
    args = parser.parse_args()

    if args.mode == "mock":
        data = make_mock_inputs()
        out_png = args.out or str(PROJECT_ROOT / "outputs" / "reports" / "three_panel" / "three_panel_mock.png")
        print("[*] 使用模拟数据生成三联图")
        fig = plot_three_panel(
            x_axis=data["x_axis"], results=data["results"], smooth=data["smooth"],
            val_loss=data.get("val_loss"), conf_matrix=data["conf_matrix"],
            classes=data["classes"], pr_data=data["pr_data"], save_path=out_png,
        )
        plt.close(fig)
    else:
        out_png = export_real_figure(model_id=args.model_id, out_png=args.out or None)
        print(f"[*] 使用真实数据(模型 {args.model_id}), 新增输出目录: "
              f"{PROJECT_ROOT / 'outputs' / 'reports' / 'three_panel' / f'model{args.model_id}'}")
    print(f"[✓] 已保存: {out_png}")


if __name__ == "__main__":
    main()
