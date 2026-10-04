"""Step 4: draw the three figures used in the article."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paths import RESULTS, FIGURES

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: illustrative newsvendor cost curve (mean demand 100, shortage 4x leftover)
rng = np.random.default_rng(7)
sig = 0.25
d = 100 * np.exp(sig * rng.standard_normal(400_000) - sig ** 2 / 2)
qs = np.arange(70, 161)
cs = np.array([(4 * np.maximum(d - q, 0) + np.maximum(q - d, 0)).mean() for q in qs])
idx = cs / cs.min() * 100
fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
ax.plot(qs, idx, color=BLUE, lw=2)
best = qs[cs.argmin()]
ax.scatter([100], [idx[qs == 100][0]], s=50, color=ORANGE, edgecolor="white", zorder=3)
ax.scatter([best], [100], s=50, color=BLUE, edgecolor="white", zorder=3)
ax.annotate(f"Stock the average (100)\ncost {idx[qs == 100][0] - 100:.0f}% above the best plan",
            xy=(100, idx[qs == 100][0]), xytext=(104, 162), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate(f"Best plan: {best} units\n(80th percentile of demand)", xy=(best, 100), xytext=(128, 88),
            fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("Units stocked (average demand = 100; a shortage costs 4x a leftover unit)")
ax.set_ylabel("Expected cost (best = 100)")
ax.set_ylim(82, 185)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIGURES / "newsvendor_cost_curve.png", facecolor="white")
plt.close(fig)

# Figure 2: forecast error vs decision cost
R = pd.read_csv(RESULTS / "model_results.csv")
label_pos = {  # (x, y, horizontal alignment) for each label
    "Machine learning forecast": (1.575, 104.3, "center"), "28-day average": (1.66, 95.5, "left"),
    "Weekday average": (1.72, 104, "left"), "Weekday rule": (1.98, 92.8, "center"),
    "Machine learning forecast, converted": (1.74, 88.0, "center"),
    "Machine learning, trained on 80th pct": (2.12, 86.8, "center"),
}
fig, ax = plt.subplots(figsize=(8, 3.8), dpi=300)
for _, r in R.iterrows():
    high = r.stocks_to == "80th percentile"
    ax.scatter(r.forecast_error_mae, r.cost_index, s=70, color=ORANGE if high else BLUE,
               edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(r.approach, xy=(r.forecast_error_mae, r.cost_index), xytext=label_pos[r.approach][:2],
                fontsize=8.5, color=INK, ha=label_pos[r.approach][2], va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkB=5))
ax.scatter([], [], color=BLUE, s=50, label="Stocks to the average")
ax.scatter([], [], color=ORANGE, s=50, label="Stocks to the 80th percentile")
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
ax.set_xlabel("Average forecast error (units per product-day; lower = more accurate)")
ax.set_ylabel("Decision cost (ML forecast = 100)")
ax.set_xlim(1.45, 2.3)
ax.set_ylim(78, 106)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIGURES / "accuracy_vs_cost.png", facecolor="white")
plt.close(fig)

# Figure 3: savings by cost ratio
S = pd.read_csv(RESULTS / "cost_ratio_sensitivity.csv")
fig, ax = plt.subplots(figsize=(8, 2.6), dpi=300)
x = np.arange(len(S))
ax.bar(x, S.saving * 100, width=0.5, color=BLUE)
for i, v in enumerate(S.saving * 100):
    ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10, color=INK, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Shortage costs 2x a leftover", "4x", "9x"])
ax.set_ylabel("Cost reduction")
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_ylim(0, 45)
fig.tight_layout()
fig.savefig(FIGURES / "savings_by_cost_ratio.png", facecolor="white")
plt.close(fig)
print("figures written")
