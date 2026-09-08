"""
=====================================================================
merge_wrong_cases.py —— 将 wrong_cases 每个模型文件夹内的逐个错检 CSV
合并为一个 CSV (每个文件夹 -> 一个 csv)
=====================================================================
输入: outputs/reports/new_env_test/wrong_cases/<模型名>/ 下所有错检样本 CSV
输出: outputs/reports/new_env_test/wrong_cases/<模型名>/<模型名>.csv
      内容仅含: sample_id, filename, true_label, <模型名>_prob
=====================================================================
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

WRONG_ROOT = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "wrong_cases"


def main() -> None:
    if not WRONG_ROOT.exists():
        raise FileNotFoundError(f"未找到目录: {WRONG_ROOT}")

    folders = sorted([p for p in WRONG_ROOT.iterdir() if p.is_dir()])
    if not folders:
        raise FileNotFoundError(f"{WRONG_ROOT} 下无模型子文件夹")

    for folder in folders:
        model_name = folder.name  # 如 1_best_model / ensemble_vote
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            print(f"[跳过] {folder} 无 csv")
            continue

        # 每个错检样本 csv 都是单行, 只取所需列并重命名为 prob
        rows = []
        for f in csvs:
            if f.stem == model_name:  # 跳过上次合并产生的输出文件
                continue
            df = pd.read_csv(f)
            prob_col = f"{model_name}_prob"
            # 兼容: 若文件没有该列则用任意 *_prob
            col = prob_col if prob_col in df.columns else [c for c in df.columns if c.endswith("_prob")][0]
            sub = df[["sample_id", "filename", "true_label", col]].copy()
            sub = sub.rename(columns={col: prob_col})
            rows.append(sub)

        merged = pd.concat(rows, ignore_index=True)
        # 同一模型文件夹内样本不重复, 仍做一次安全去重并按 sample_id 排序
        merged = merged.drop_duplicates(subset=["sample_id"]).sort_values("sample_id")
        merged["sample_id"] = merged["sample_id"].astype(int)

        out = folder / f"{model_name}.csv"
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[完成] {out}  (共 {len(merged)} 行, 列: {list(merged.columns)})")


if __name__ == "__main__":
    main()
