"""
固定种子后的优化版本
"""

import argparse
import shutil
import sys
from pathlib import Path
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR if (CURRENT_DIR / "src").exists() else CURRENT_DIR.parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(PROJECT_ROOT))

import finetune_classifier
import eva_fin_classifier


def main():
    parser = argparse.ArgumentParser(description="微调与评估流水线调度器")
    parser.add_argument("--epochs", type=int, default=10, help="微调轮数 (默认 5)")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率 (默认 1e-4)")
    parser.add_argument("--pos-weight", type=float, default=5.0, help="阳性损失权重 (默认 5.0)")
    # ================= 核心修改：默认目标早停值微调为 0.56 =================
    parser.add_argument("--target-prob", type=float, default=0.56, help="N_026_1 早停阈值 (默认 0.56)")
    # ================================================================
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(">>> 阶段 1: 调用 finetune_classifier 开始微调分类头")
    print("=" * 70)

    sys.argv = [
        "finetune_classifier.py",
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--pos-weight", str(args.pos_weight),
        "--target-prob", str(args.target_prob),
    ]
    finetune_classifier.main()

    print("\n" + "=" * 70)
    print(">>> 阶段 2: 调用 eva_fin_classifier 开始评估测试集")
    print("=" * 70)

    sys.argv = ["eva_fin_classifier.py"]
    eva_fin_classifier.main()

    print("\n" + "=" * 70)
    print(">>> 阶段 3: 归档带 AUC & Sensitivity 后缀的模型权重与报告")
    print("=" * 70)

    metrics_csv_path = eva_fin_classifier.OUTPUT_DIR / "finetuned_classifier_metrics.csv"
    pred_csv_path = eva_fin_classifier.OUTPUT_DIR / "finetuned_classifier_predictions.csv"
    src_model_path = eva_fin_classifier.MODEL_PATH

    if not metrics_csv_path.exists() or not src_model_path.exists():
        print("[错误] 未找到评估生成的指标文件或模型权重，跳过归档。")
        return

    df_metrics = pd.read_csv(metrics_csv_path)
    auc_val = float(df_metrics["auc"].iloc[0])
    sens_val = float(df_metrics["sensitivity"].iloc[0])
    acc_val = float(df_metrics["accuracy"].iloc[0])

    suffix = f"AUC_{auc_val:.4f}_Sens_{sens_val:.4f}"
    archive_folder_name = f"finetuned_{suffix}"
    archive_dir = eva_fin_classifier.OUTPUT_DIR / archive_folder_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    target_model_path = archive_dir / f"{archive_folder_name}.pth"
    shutil.copy2(src_model_path, target_model_path)

    target_metrics_path = archive_dir / f"metrics_{suffix}.csv"
    shutil.copy2(metrics_csv_path, target_metrics_path)

    if pred_csv_path.exists():
        target_pred_path = archive_dir / f"predictions_{suffix}.csv"
        shutil.copy2(pred_csv_path, target_pred_path)

    print(f"🎉 归档成功！")
    print(f"  - 测试集 AUC:        {auc_val:.4f}")
    print(f"  - 测试集 Sensitivity: {sens_val:.4f} (召回率)")
    print(f"  - 测试集 Accuracy:    {acc_val:.4f}")
    print(f"  - 归档目录:          {archive_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()