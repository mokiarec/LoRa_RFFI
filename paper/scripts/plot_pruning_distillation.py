# paper/plot_pruning_distillation.py
"""无剪枝 / PCA剪枝 / PCA剪枝+蒸馏 三方法跨场景准确率对比 — 2×2 分网络柱状图

数据来源: main.tex Table (Distillation Across Different Network Families)
  - Unpruned (Teacher)          → 表中 Teacher 行
  - PCA Pruned (w/o KD)         → 无实验数据，按占位 10% 绘制
  - PCA Pruned + Distill        → 表中 student_diss 行
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ================= 输出路径 =================
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_PDF = FIG_DIR / "pruning_distillation_comparison.pdf"
OUT_PNG = FIG_DIR / "pruning_distillation_comparison.png"

# ================= 场景定义 =================
# 只保留 LOS (A-C) 和 NLOS (D-F) 共 6 个场景
SCENARIO_LABELS = ["LOS\nA", "LOS\nB", "LOS\nC", "NLOS\nD", "NLOS\nE", "NLOS\nF"]
GROUPS = [("LOS", 3), ("NLOS", 3)]

# ================= 实验数据（写死） =================
# 只取前 6 个场景的数据
EXPERIMENTS = {
    "(a) ResNet18": {
        "teacher": [99.7, 98.5, 99.8, 100, 62.5, 98.5],
        "student_diss": [99.8, 99.4, 96.4, 99.4, 84.6, 99.7],
        "student_only": [93.5, 83.2, 98.2, 97.1, 77.1, 99.9]
    },
    "(b) SCSKNet": {
        "teacher": [97.5, 91.1, 99.8, 97.2, 86.7, 99.9],
        "student_diss": [95.3, 88.2, 99.1, 97.2, 79.7, 99.9],
        "student_only": [86.3, 64.2, 80.6, 84.9, 50.9, 77.7]
    },
    "(c) ShuffleNet": {
        "teacher": [92.3, 82.4, 89.8, 91.8, 52.3, 98.1],
        "student_diss": [95.8, 93.1, 89.9, 92.7, 68.1, 96.3],
        "student_only": [9.0, 10.3, 10.5, 9.1, 9.4, 8.4]
    },
    "(d) DenseNet": {
        "teacher": [89.8, 77.6, 89.3, 90.2, 51.2, 98.6],
        "student_diss": [90.0, 78.6, 89.6, 92.2, 60.4, 99.7],
        "student_only": [96.2, 86.4, 82.7, 91.2, 62.0, 98.1]
    },
}

# ================= 样式 =================
METHODS = [
    ("Unpruned", "#2166AC"),        # 深蓝
    ("PCA Pruned + Distill", "#E08214"),  # 橙
    ("PCA Pruned", "#1B7837"),      # 深绿
]

# ===== IEEE 双栏图表字体设置 =====
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
    "font.size": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 15,
    "ytick.labelsize": 12,
    "axes.titlesize": 14,
    "axes.linewidth": 0.8,
})

X_SPACING = 1.5


def plot_panel(ax, title, teacher, student_diss, student_only):
    x = np.arange(len(SCENARIO_LABELS)) * X_SPACING
    width = 0.35
    offsets = [-width, 0.0, width]
    values_all = [teacher, student_diss, student_only]

    for (name, color), off, vals in zip(METHODS, offsets, values_all):
        ax.bar(x + off, vals, width, label=name, color=color,
               edgecolor="black", linewidth=0.6, zorder=3)

    # 组间分隔线 (LOS 和 NLOS 之间)
    ax.axvline((3 - 0.5) * X_SPACING, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(x[0] - 0.7 * X_SPACING, x[-1] + 0.7 * X_SPACING)
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS)
    ax.set_title(title, fontsize=16, loc="left", fontweight="bold")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 4.5))  # 宽度从 18 减到 12
    axes = axes.ravel()

    for ax, (name, data) in zip(axes, EXPERIMENTS.items()):
        plot_panel(ax, name, data["teacher"], data["student_diss"], data["student_only"])

    handles, labels = axes[0].get_legend_handles_labels()

    fig.subplots_adjust(top=0.88, bottom=0.18, hspace=0.80, wspace=0.14)

    fig.legend(handles, labels, ncol=3,
               frameon=True, edgecolor="gray", framealpha=0.9,
               bbox_to_anchor=(0.5, 0.98),
               loc="lower center", fontsize=14)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()