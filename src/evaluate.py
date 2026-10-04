"""Step 3: turn forecasts into stocking decisions and cost them.

Each forecast is treated as the stock level for that product-day.
Shortage cost = 0.4 x price per unit; leftover cost = 0.1 x price per unit
(a 4:1 ratio, so the cost-minimising stock level is the 80th percentile).
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from paths import WORK, RESULTS

F = pd.read_parquet(WORK / "forecasts.parquet")
y, price = F.y.values, F.sell_price.values


def cost(stock, short=0.4, over=0.1, y=y, price=price):
    return (price * (short * np.maximum(y - stock, 0) + over * np.maximum(stock - y, 0))).sum()


APPROACHES = {
    "Machine learning forecast":                 ("lgb_mean", "average", None),
    "28-day average":                            ("ma28", "average", None),
    "Weekday average":                           ("wd8", "average", None),
    "Machine learning forecast, converted":      ("lgb_mean", "80th percentile", "poisson"),
    "Weekday rule":                              ("wd8_q80", "80th percentile", None),
    "Machine learning, trained on 80th pct":     ("lgb_q80", "80th percentile", None),
}
rows = []
for name, (col, target, convert) in APPROACHES.items():
    f = F[col].values
    stock = poisson.ppf(0.8, np.maximum(f, 1e-9)) if convert else np.round(f)
    reported = stock if convert else f
    rows.append(dict(approach=name, stocks_to=target,
                     forecast_error_mae=np.abs(y - reported).mean(),
                     mean_error_bias=(reported - y).mean(),
                     decision_cost=cost(stock),
                     sales_covered=np.minimum(stock, y).sum() / y.sum()))
R = pd.DataFrame(rows)
R["cost_index"] = 100 * R.decision_cost / R.loc[R.approach == "Machine learning forecast", "decision_cost"].item()
R.to_csv(RESULTS / "model_results.csv", index=False)
print(R.round(3).to_string(index=False))

# Sensitivity: stock the ML forecast at the matching percentile for each cost ratio
sens = []
for short, over in [(0.2, 0.1), (0.4, 0.1), (0.9, 0.1)]:
    q = short / (short + over)
    base = cost(np.round(F.lgb_mean.values), short, over)
    tuned = cost(poisson.ppf(q, np.maximum(F.lgb_mean.values, 1e-9)), short, over)
    sens.append(dict(cost_ratio=f"{short / over:.0f}:1", percentile=round(q, 3), saving=1 - tuned / base))
S = pd.DataFrame(sens)
S.to_csv(RESULTS / "cost_ratio_sensitivity.csv", index=False)
print(S.round(3).to_string(index=False))

# Stability by window
for T, G in F.groupby("origin"):
    c = lambda s: cost(s, y=G.y.values, price=G.sell_price.values)
    base = c(np.round(G.lgb_mean.values))
    print(f"window after day {T}: quantile-model saving {1 - c(np.round(G.lgb_q80.values)) / base:.3f}")
