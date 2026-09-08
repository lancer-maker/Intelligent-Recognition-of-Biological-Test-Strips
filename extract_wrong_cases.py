"""
=====================================================================
extract_wrong_cases.py —— 提取错检样本，按模型序号生成逐个 CSV
=====================================================================
数据来源:
  - outputs/reports/new_env_test/predictions/ 下所有 *preds.csv
    (1~5_best_model + ensemble_vote), 每行对应一个测试样本
  - data/test/labels/labels.csv: 原图信息(filename/label/item_id)
    与 sample_id 按行一一对应(sample_id = 1-based 行号)

错检定义: pred_label != test_labels

输出:
  outputs/reports/new_env_test/wrong_cases/<模型名>/<sample_id>_<原图名>.csv
  每个错检样本生成一个 CSV, 内容为该样本的
  原图信息(sample_id/filename/true_label/item_id) +
  全部 6 个来源(1~5 单模型 + 集成)对该样本的 prob/pred/对错汇总
=====================================================================
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

PRED_DIR = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "predictions"
LABELS_CSV = PROJECT_ROOT / "data" / "test" / "labels" / "labels.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "wrong_cases"


def load_pred_sources() -> list:
    """读取所有 *preds.csv, 返回 [(source_name, df)] 按自然顺序。"""
    files = sorted(PRED_DIR.glob("*preds.csv"), key=lambda p: p.name)
    sources = []
    for f in files:
        name = f.name.replace("_preds.csv", "")  # 如 1_best_model / ensemble_vote
        df = pd.read_csv(f)
        # 容错: sample_id 缺失时用 1-based 行号补全
        if "sample_id" not in df.columns:
            df["sample_id"] = df.index + 1
        df["pred_label"] = df["pred_label"].astype(int)
        sources.append((name, df))
    return sources


def load_label_map() -> pd.DataFrame:
    """读取 labels.csv, 附加 1-based sample_id 列。"""
    df = pd.read_csv(LABELS_CSV)
    df.insert(0, "sample_id", df.index + 1)
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={"label": "true_label", "filename": "filename"})
    if "item_id" not in df.columns:
        df["item_id"] = ""
    df["item_id"] = df["item_id"].fillna("")
    return df


def main() -> None:
    sources = load_pred_sources()
    labels = load_label_map()

    # 构建按 sample_id 的预测信息: {sample_id: {source: (prob, pred)}}
    pred_by_id = {}
    for name, df in sources:
        for _, row in df.iterrows():
            sid = int(row["sample_id"])
            pred_by_id.setdefault(sid, {})[name] = (
                float(row["test_probs"]),
                int(row["pred_label"]),
            )

    source_names = [n for n, _ in sources]
    n_created = 0
    total_wrong = 0

    # 每个来源(模型)各建一个子目录, 只放该模型判错的样本
    for name, _ in sources:
        folder = OUTPUT_ROOT / name
        folder.mkdir(parents=True, exist_ok=True)

        wrong_ids = []
        for sid, info in pred_by_id.items():
            _, pred = info[name]
            true_label = int(labels.loc[labels["sample_id"] == sid, "true_label"].iloc[0])
            if pred != true_label:
                wrong_ids.append(sid)

        for sid in sorted(wrong_ids):
            rec = labels.loc[labels["sample_id"] == sid].iloc[0]
            stem = Path(str(rec["filename"])).stem  # 如 N_002_0
            out_csv = folder / f"{sid:03d}_{stem}.csv"

            # 组装单行记录: 原图信息 + 各来源预测汇总
            row_out = {
                "sample_id": sid,
                "filename": rec["filename"],
                "true_label": int(rec["true_label"]),
                "item_id": rec["item_id"],
            }
            wrong_in = []
            for src in source_names:
                pprob, ppred = pred_by_id[sid][src]
                result = "correct" if ppred == int(rec["true_label"]) else "wrong"
                row_out[f"{src}_prob"] = round(pprob, 6)
                row_out[f"{src}_pred"] = int(ppred)
                row_out[f"{src}_result"] = result
                if result == "wrong":
                    wrong_in.append(src)
            row_out["wrong_in"] = ";".join(wrong_in)
            row_out["n_sources_wrong"] = len(wrong_in)

            pd.DataFrame([row_out]).to_csv(
                out_csv, index=False, encoding="utf-8-sig"
            )
            n_created += 1

        total_wrong += len(wrong_ids)
        print(f"[{name}] 错检样本 {len(wrong_ids)} 个 -> {folder}")

    print(f"\n共生成 {n_created} 个错检样本 CSV (错检样本去重后 {total_wrong} 次判定)")


if __name__ == "__main__":
    main()
