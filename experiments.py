"""Эксперименты для отчёта: scatter-графики и heatmap корреляций."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

project_dir = Path(__file__).resolve().parent
figures_dir = project_dir / "figures"
figures_dir.mkdir(exist_ok=True)

df = pd.read_csv(project_dir / "brainsize.txt", sep="\t", na_values="NA")
features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, f in zip(axes, features):
    sns.scatterplot(data=df, x=f, y="MRI_Count", hue="Gender", ax=ax)
    ax.set_title(f"{f} vs MRI_Count")
plt.tight_layout()
plt.savefig(figures_dir / "scatter.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, g in zip(axes, ["Female", "Male"]):
    corr = df[df.Gender == g][features + ["MRI_Count"]].corr(method="pearson")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Корреляции Пирсона: {g}")
plt.tight_layout()
plt.savefig(figures_dir / "heatmap.png", dpi=120)
