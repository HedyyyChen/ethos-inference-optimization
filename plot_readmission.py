#!/usr/bin/env python3
"""Plot readmission results (ROC and PR) for a results folder.
Modified to add AUPRC info to the PR curve.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from ethos.metrics import (
    process_readmission_results,
    compute_gaussian_metrics,
    compute_basic_metrics,
    print_auc_roc_plot,
)
from ethos.constants import ADMISSION_STOKEN


def plot_pr_curve_with_info(gauss_dict, y_true, y_pred, title, save_path):
    """
    绘制 PR 曲线，并添加 AUPRC、置信区间和样本信息。
    """
    # 1. 计算基础 AUPRC
    basic_auprc = average_precision_score(y_true, y_pred)
    
    # 2. 从高斯拟合结果中获取 AUPRC 的均值和置信区间
    # 注意：`compute_gaussian_metrics` 通常也会计算 PR 相关的指标
    gauss_auprc = gauss_dict.get("auprc", basic_auprc) # 如果有就用，没有就回退
    ci_lower = gauss_dict.get("auprc_ci", [basic_auprc, basic_auprc])[0]
    ci_upper = gauss_dict.get("auprc_ci", [basic_auprc, basic_auprc])[1]
    
    # 3. 绘图
    plt.figure(figsize=(7, 6))
    plt.plot(gauss_dict["recall_values"], gauss_dict["precision_values"], color="darkorange", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    
    # 4. 准备文本信息
    n_pos = np.sum(y_true)
    n_total = len(y_true)
    text_lines = [
        f"Gaussian AUPRC: {gauss_auprc:.3f}",
        f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
        f"N: {n_total:,} ({n_pos/n_total:.1%} positives)"
    ]
    textstr = '\n'.join(text_lines)
    
    # 5. 添加文本框（位置可以调整）
    plt.gcf().text(0.65, 0.25, textstr, fontsize=12, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Results folder name under PROJECT_ROOT/results or full path")
    p.add_argument("--outdir", default="plots", help="Output directory for plots")
    p.add_argument("--period", type=float, default=30 / 365.25, help="Readmission period in years (default 30 days)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = process_readmission_results(args.results, ADMISSION_STOKEN, args.period)

    y_true = df.expected
    y_pred = df.actual

    basic = compute_basic_metrics(y_true, y_pred)
    gauss = compute_gaussian_metrics(y_true, y_pred)

    # --- 绘制 ROC 图 (官方原有逻辑) ---
    plt.figure(figsize=(7, 6))
    print_auc_roc_plot(basic, gauss, title=f"Readmission ROC: {args.results}")
    roc_path = outdir / f"{args.results}_roc.png"
    plt.savefig(roc_path, bbox_inches="tight", dpi=200)
    plt.close()

    # --- 绘制 PR 图 (使用我们新创建的函数) ---
    pr_path = outdir / f"{args.results}_pr.png"
    plot_pr_curve_with_info(gauss, y_true, y_pred, f"Readmission PR: {args.results}", pr_path)

    print(f"Saved ROC -> {roc_path}")
    print(f"Saved PR  -> {pr_path}")


if __name__ == "__main__":
    main()