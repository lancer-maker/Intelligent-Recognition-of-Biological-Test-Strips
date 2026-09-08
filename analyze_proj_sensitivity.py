"""
=====================================================================
analyze_proj_sensitivity.py —— 投影流关注"变化曲线"还是"具体取值"?
=====================================================================
背景:
  投影输入在送入 ProjStream 前已做 min-max 归一化 + 去均值中心化 +
  线性 detrend (见 data/utils.py normalize_projection_baseline), 因此
  整体亮度/直流电平理论上已被移除。本脚本在真实模型上做受控扰动实验,
  量化 ProjStream 对信号不同属性的依赖程度:

    条件类别           含义
    ---------------------------------------------------------------
    dc_offset         叠加直流(整体电平) -> 具体取值(亮度水平)
    gain              乘性幅度缩放       -> 具体取值(峰值幅度/强度)
    shape_only        每通道按自身 std 归一 -> 只保留"变化曲线"形状, 去掉绝对幅度
    flip              左右镜像           -> 变化曲线的方向/位置
    block_shuffle     块重排(保留取值,打乱位置) -> 变化曲线的空间顺序
    noise             加高斯噪声          -> 对精细波形的敏感度

  若 dc/gain 几乎不影响输出, 而 shape_only 维持原精度 -> 模型依赖"变化曲线";
  若 gain 强烈改变输出且 shape_only 精度大幅下降   -> 模型依赖"具体取值"(幅度)。

输出: outputs/reports/new_env_test/proj_sensitivity.csv
=====================================================================
"""
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 new_env_train 的数据读取辅助 (保证与训练同一数据管线)
from new_env_train import prepare_labels, read_csv_directly
from src.data.dataset import TestStripDataset
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

# ===================== 配置区 (不采用终端输入) =====================
CONFIG_PATH = PROJECT_ROOT / "configs" / "main_config.yaml"   # 与 new_env_train 同一份配置
EVAL_CSV = "data/test/labels/labels.csv"                       # 待分析数据集标注 (相对根目录)
EVAL_RAW_DIR = PROJECT_ROOT / "data" / "test" / "raw"         # 待分析数据集图片目录
MODEL_IDS: List[int] = [1, 2, 3, 4, 5]                         # new_env_train 产出的模型编号
OUT_CSV = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "proj_sensitivity.csv"
SEED = 0
rng = np.random.default_rng(SEED)
# ================================================================


def _load_config() -> dict:
    """读取主配置 (同 new_env_train.main 的 yaml 读取方式)。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> DualStreamStripNet:
    """按 main_config 构建双流网络 (参数与 new_env_train 完全一致)。"""
    return DualStreamStripNet(
        pretrained_2d=bool(cfg["model"]["pretrained"]),
        feature_dim=int(cfg["model"]["feature_dim"]),
        dropout_rate=float(cfg["model"]["dropout_rate"]),
    )


def build_eval_dataset(cfg: dict):
    """按 new_env_train 方式读取数据集并构造确定性 Dataset。

    - 关闭投影随机增强 (projection_transforms=False) -> 与 new_env_test 默认一致、可复现;
    - 图像侧使用确定性 val 预处理 (确定性 CLAHE + 归一化)。
    """
    df = read_csv_directly(str(EVAL_CSV))
    prepared = prepare_labels(df, EVAL_RAW_DIR)
    ds = TestStripDataset(
        image_paths=prepared["filename"].tolist(),
        labels=prepared["label"].tolist(),
        transforms=get_val_transforms(),
        projection_transforms=False,
        standard_dpi_size=tuple(cfg["data"]["standard_dpi_size"]),
        proj_length=int(cfg["data"]["proj_length"]),
    )
    return ds, prepared["label"].astype(int).to_numpy()


# ---------------- 扰动工具 ----------------
def build_variants(x: torch.Tensor):
    """输入 x: (2,512) 已去基线。返回 {名称: 变形后的 (2,512)}。"""
    variants = {"baseline": x.clone()}
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)

    # 1) 具体取值-整体电平: 叠加直流
    for c in (0.05, 0.15, -0.15):
        variants[f"dc{c:+.2f}"] = x + c

    # 2) 具体取值-幅度: 乘性缩放
    for g in (0.3, 0.6, 1.5, 3.0):
        variants[f"gain{g:.1f}"] = x * g

    # 3) 变化曲线-仅形状(去掉绝对幅度)
    variants["shape_only"] = x / std

    # 4) 变化曲线-方向/位置: 左右镜像
    variants["flip"] = torch.flip(x, dims=[-1])

    # 5) 变化曲线-空间顺序(保留取值打乱位置): 分8块重排
    n_pts = x.shape[-1]
    n_blocks = 8
    blk = n_pts // n_blocks
    order = rng.permutation(n_blocks)
    x_shuf = torch.empty_like(x)
    for ch in range(x.shape[0]):
        blocks = [x[ch, i * blk:(i + 1) * blk].clone() for i in range(n_blocks)]
        x_shuf[ch] = torch.cat([blocks[o] for o in order], dim=0)
    variants["block_shuffle"] = x_shuf

    # 6) 精细波形敏感性: 高斯噪声
    torch.manual_seed(SEED)
    variants["noise_0.02"] = x + torch.randn_like(x) * 0.02

    return variants


# ---------------- 主流程 ----------------
def main():
    cfg = _load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 配置: {CONFIG_PATH.name}")

    # 按 new_env_train 方式读取数据集 (确定性预处理)
    ds, labels = build_eval_dataset(cfg)
    names = [Path(p).name for p in ds.image_paths]
    print(f"分析数据集: {len(names)} 张 (阳性 {int(labels.sum())})")

    rows = []
    for mid in MODEL_IDS:
        ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "Tweak" / str(mid) / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[跳过] 无权重: {ckpt_path}")
            continue
        model = build_model(cfg).to(device)
        model = load_model_state(model, str(ckpt_path))
        model.eval()

        # 缓存每个样本的融合/投影/图像基线与各扰动结果
        per_cond = {k: [] for k in
                    ["baseline", "dc+0.05", "dc+0.15", "dc-0.15",
                     "gain0.3", "gain0.6", "gain1.5", "gain3.0",
                     "shape_only", "flip", "block_shuffle", "noise_0.02"]}
        accs = {k: [] for k in per_cond}
        proj_only_base, image_only_base, fusion_base = [], [], []

        with torch.no_grad():
            for i in range(len(ds)):
                y = int(labels[i])
                img_t, proj_t, _ = ds[i]
                img_t = img_t.unsqueeze(0).to(device)            # (1,3,505,220)
                x0 = proj_t.to(device)                           # (C, L) 信号通道 (dataset 输出)
                img_feats = model.image_stream(img_t)            # (1, feature_dim)
                zero = torch.zeros_like(img_feats)

                def prob_fusion(x):
                    pf = model.proj_stream(x.unsqueeze(0))
                    return torch.sigmoid(model.classifier(
                        torch.cat([img_feats, pf], dim=1))).item()

                def prob_proj(x):
                    pf = model.proj_stream(x.unsqueeze(0))
                    return torch.sigmoid(model.classifier(
                        torch.cat([zero, pf], dim=1))).item()

                # 单流基准
                proj_only_base.append(prob_proj(x0))
                image_only_base.append(torch.sigmoid(model.classifier(
                    torch.cat([img_feats, zero], dim=1))).item())
                fusion_base.append(prob_fusion(x0))

                for name, xv in build_variants(x0).items():
                    p = prob_fusion(xv)
                    per_cond[name].append(p)
                    accs[name].append(int(p >= 0.5) == y)

        n = len(fusion_base)
        base_fusion = np.asarray(fusion_base)
        base_acc = np.mean(np.asarray(accs["baseline"]))
        ponly = np.asarray(proj_only_base)
        ionly = np.asarray(image_only_base)

        print(f"\n=== 模型 {mid} ===")
        print(f"  fusion_acc={base_acc:.3f} | image_only_acc="
              f"{np.mean((ionly>=0.5)==labels):.3f} | proj_only_acc="
              f"{np.mean((ponly>=0.5)==labels):.3f}")

        row_base = {
            "model": f"Tweak/{mid}", "condition": "baseline",
            "fusion_acc": round(base_acc, 4),
            "proj_only_acc": round(float(np.mean((ponly >= 0.5) == labels)), 4),
            "image_only_acc": round(float(np.mean((ionly >= 0.5) == labels)), 4),
            "mean_abs_delta": 0.0, "flip_rate": 0.0,
        }
        rows.append(row_base)

        for cond in per_cond:
            if cond == "baseline":
                continue
            arr = np.asarray(per_cond[cond])
            mean_delta = float(np.mean(np.abs(arr - base_fusion)))
            flip_rate = float(np.mean((arr >= 0.5) != (base_fusion >= 0.5)))
            acc = float(np.mean(accs[cond]))
            print(f"  {cond:<14s} mean|Δ|={mean_delta:.4f}  flip={flip_rate:.3f}  acc={acc:.3f}")
            rows.append({
                "model": f"Tweak/{mid}", "condition": cond,
                "fusion_acc": round(acc, 4),
                "proj_only_acc": None, "image_only_acc": None,
                "mean_abs_delta": round(mean_delta, 4),
                "flip_rate": round(flip_rate, 4),
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
