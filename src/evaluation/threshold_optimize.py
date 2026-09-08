# 约登指数最佳阈值搜索与交叉验证汇总统计模块
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Tuple, Dict, List, Optional
from .metrics import compute_metrics


def calculate_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    基于 ROC 曲线通过约登指数最大化寻优最佳决策阈值。
    约登指数 (Youden Index) = 灵敏度 + 特异度 - 1
    
    Args:
        y_true: 真实标签数组 (0 或 1)
        y_prob: 模型预测阳性概率数组 [0.0, 1.0]
        
    Returns:
        Tuple[float, float, float, float, float]: 
        (最佳阈值, 该阈值下的灵敏度, 该阈值下的特异度, 该阈值下的准确率, AUC)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    # 边缘情况处理：若验证集中只有单一类别（例如 8 张全阴性或全阳性）
    if len(np.unique(y_true)) < 2:
        metrics_05 = compute_metrics(y_true, y_prob, threshold=0.5)
        return 0.5, metrics_05['sensitivity'], metrics_05['specificity'], metrics_05['accuracy'], metrics_05['auc']

    # 1. 计算 ROC 曲线采样点
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    # 2. 计算每个阈值点对应的约登指数
    youden_indices = tpr + (1.0 - fpr) - 1.0
    
    # 3. 寻找约登指数最大值索引
    best_idx = np.argmax(youden_indices)
    best_threshold = float(thresholds[best_idx])

    # 修复 sklearn 的边界处理伪影 (有时 thresholds[0] 为 inf)
    if np.isinf(best_threshold):
        best_threshold = 1.0

    # 4. 根据寻优得到的最佳阈值重新评估精确指标
    best_metrics = compute_metrics(y_true, y_prob, threshold=best_threshold)

    return (
        best_threshold,
        best_metrics['sensitivity'],
        best_metrics['specificity'],
        best_metrics['accuracy'],
        roc_auc
    )


def compute_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    基于 Bootstrap 重采样计算 AUC 的 95% 置信区间。

    对样本对有放回重采样, 每个重采样样本计算 AUC, 以 alpha 分位数作为置信区间。
    小样本 (如 LOOCV 每折仅 8 张) 会导致区间较宽, 这正是希望呈现的不确定性。

    Args:
        y_true: 真实标签数组 (0 或 1)
        y_prob: 模型预测阳性概率数组 [0.0, 1.0]
        n_bootstraps: Bootstrap 重采样次数 (默认 1000)
        alpha: 置信水平 (默认 0.95)
        seed: 随机种子, 保证可复现

    Returns:
        Tuple[float, float, float]: (bootstrap AUC 均值, 置信下限, 置信上限);
        若正负样本不足或全部重采样样本无效则返回 (None, None, None)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    # 正负样本任一缺失时无法计算 AUC
    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return None, None, None

    rng = np.random.RandomState(seed)
    n = len(y_true)
    idxs = np.arange(n)
    aucs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(idxs, size=n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        # 重采样后若某一类缺失, 该样本无法计算 AUC, 直接跳过
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yt, yp)))
        except ValueError:
            continue

    if not aucs:
        return None, None, None

    aucs = np.asarray(aucs)
    lo = float(np.percentile(aucs, (1.0 - alpha) / 2.0 * 100.0))
    hi = float(np.percentile(aucs, (1.0 + alpha) / 2.0 * 100.0))
    return float(np.mean(aucs)), lo, hi


def summarize_loocv_results(fold_results: List[Dict[str, float]]) -> pd.DataFrame:
    """
    汇总并格式化 24 折留一交叉验证 (LOOCV) 的实验指标。
    
    Args:
        fold_results: 每折评估字典组成的列表
        
    Returns:
        pd.DataFrame: 包含各折指标以及最后一行 Mean ± Std 统计总结的 DataFrame
    """
    df = pd.DataFrame(fold_results)
    
    # 计算均值与标准差
    summary_mean = df.mean(numeric_only=True)
    summary_std = df.std(numeric_only=True)

    print("\n==================== LOOCV PERFORMANCE REPORT ====================")
    print(f"AUC:         {summary_mean['auc']:.4f} ± {summary_std['auc']:.4f}")
    print(f"Sensitivity: {summary_mean['sensitivity']:.4f} ± {summary_std['sensitivity']:.4f}")
    print(f"Specificity: {summary_mean['specificity']:.4f} ± {summary_std['specificity']:.4f}")
    print(f"Accuracy:    {summary_mean['accuracy']:.4f} ± {summary_std['accuracy']:.4f}")
    print(f"Optimal TH:  {summary_mean['best_threshold']:.4f} ± {summary_std['best_threshold']:.4f}")
    # 若每折已计算 AUC 95% 置信区间, 报告跨折的平均区间
    if 'auc_ci_low' in df.columns and 'auc_ci_high' in df.columns:
        ci_low_mean = df['auc_ci_low'].mean()
        ci_high_mean = df['auc_ci_high'].mean()
        n_valid = df[['auc_ci_low', 'auc_ci_high']].dropna().shape[0]
        print(f"AUC 95% CI:  [{ci_low_mean:.4f}, {ci_high_mean:.4f}]  (跨 {n_valid}/{len(df)} 折可计算)")
        print("  -> 小样本(每折仅数张)导致区间较宽")
    print("==================================================================")

    return df