# 基础分类指标计算模块，计算准确率（Accuracy）、灵敏度（Sensitivity/召回率）、特异度（Specificity）、精准率（Precision）、F1 值与 AUC-ROC
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score
from typing import Dict, Any


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    根据给定阈值计算二分类各项核心指标。
    
    Args:
        y_true: 真实标签数组 (0 或 1)
        y_prob: 模型预测阳性概率数组 [0.0, 1.0]
        threshold: 决策阈值，大于等于该阈值判定为阳性 (默认 0.5)
        
    Returns:
        Dict[str, Any]: 包含各项指标数值的字典
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    
    # 概率转硬分类二值 (0 或 1)
    y_pred = (y_prob >= threshold).astype(int)

    # 1. 计算 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # 2. 核心临床指标
    # 灵敏度 (Sensitivity / Recall): 阳性不漏诊率
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # 特异度 (Specificity): 阴性不误诊率
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # 精准率 (Precision)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # 准确率 (Accuracy)
    acc = accuracy_score(y_true, y_pred)
    # F1 Score
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 3. AUC 计算 (处理验证集中可能只有单一类别的极端情况)
    try:
        if len(np.unique(y_true)) > 1:
            auc_val = float(roc_auc_score(y_true, y_prob))
        else:
            auc_val = 0.5 # 只有一个类别时 AUC 无法计算，设为默认 0.5
    except ValueError:
        auc_val = 0.5

    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "auc": auc_val,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }