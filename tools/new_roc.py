import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    average_precision_score,  # 计算AP
    precision_recall_curve, f1_score  # 计算精确率、召回率、F1
)

def cal_metric(target, predicted, show=False):
    # -------------------------- 原有指标计算（保留不变） --------------------------
    fpr, tpr, thresholds = roc_curve(target, predicted)
    _tpr = tpr
    _fpr = fpr
    tpr = tpr.reshape((tpr.shape[0], 1))
    fpr = fpr.reshape((fpr.shape[0], 1))
    scale = np.arange(0, 1, 0.00000001)
    function = interpolate.interp1d(_fpr, _tpr)
    y = function(scale)
    znew = abs(scale + y - 1)
    eer = scale[np.argmin(znew)]
    
    FPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    TPRs = {"TPR@FPR=10E-2": 0.01, "TPR@FPR=10E-3": 0.001, "TPR@FPR=10E-4": 0.0001}
    for i, (key, value) in enumerate(FPRs.items()):
        index = np.argwhere(scale == value)
        score = y[index]
        TPRs[key] = float(np.squeeze(score)) if len(score) > 0 else 0.0  # 避免索引为空
    
    auc = roc_auc_score(target, predicted)
    
    # -------------------------- 新增 AP 和 F1 计算 --------------------------
    # 1. 计算 AP（平均精度，适用于二分类/多分类，这里是二分类）
    ap = average_precision_score(target, predicted)  # 直接调用，无需额外处理
    
    # 2. 计算 F1 分数（需要先确定阈值，这里用两种常用方式）
    # 方式1：用“精确率=召回率”的阈值（对应EER的逻辑，更贴合任务）
    precision, recall, _ = precision_recall_curve(target, predicted)
    # 找到“精确率最接近召回率”的阈值对应的F1
    f1_list = 2 * (precision * recall) / (precision + recall + 1e-8)  # 加1e-8避免除零
    best_f1 = np.max(f1_list)  # 取最大F1值（最常用）
    
    return eer, TPRs, auc, {'x': scale, 'y': y}, ap, best_f1

# -------------------------- 测试代码（验证是否能运行） --------------------------
if __name__ == "__main__":
    # 模拟二分类数据（替换成你的真实target和predicted）
    target = np.random.randint(0, 2, size=1000)  # 真实标签（0=负类，1=正类）
    predicted = np.random.rand(1000) * 0.3 + target * 0.7  # 预测分数（正类分数更高）
    
    # 调用函数
    eer, tprs, auc, curve_data, ap, best_f1 = cal_metric(target, predicted, show=True)
    
    # 打印结果
    print(f"EER: {eer:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")  # 新增
    print(f"最优F1（精确率=召回率对应）: {best_f1:.4f}")  # 新增
    # print(f"阈值0.5对应的F1: {f1_05:.4f}")  # 新增
    for key, val in tprs.items():
        print(f"{key}: {val:.4f}")