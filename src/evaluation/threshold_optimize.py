# 约登指数最佳阈值搜索与交叉验证汇总统计模块
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from typing import Tuple, Dict, List
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
    print("==================================================================")
    
    return df