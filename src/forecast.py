"""Step 2: produce forecasts for three rolling 28-day windows.

Point forecasts: 28-day average, weekday average (last 8 same weekdays),
LightGBM with a Tweedie objective.
80th-percentile forecasts: weekday empirical 80th percentile and
LightGBM trained on the pinball (quantile) loss with alpha = 0.8.
All model features are lagged at least 28 days, so no information from
inside a forecast window is used.
"""
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from paths import WORK

ORIGINS = [1857, 1885, 1913]  # last day of history before each window
HORIZON = 28
TRAIN_DAYS = 540

L = pd.read_parquet(WORK / "foods_panel.parquet").sort_values(["id", "dn"]).reset_index(drop=True)
g = L.groupby("id")["y"]
for k in [28, 35, 42, 49, 56]:
    L[f"lag{k}"] = g.shift(k)
s28 = g.shift(28)
for w in [7, 28, 56]:
    L[f"rm{w}"] = s28.groupby(L.id).transform(lambda x, w=w: x.rolling(w).mean())
L["rwd4"] = L[["lag28", "lag35", "lag42", "lag49"]].mean(axis=1)
L["pmax"] = L.groupby("id").sell_price.transform("max")
L["prel"] = L.sell_price / L.pmax
L["store_c"] = L.store_id.astype("category").cat.codes
L["dept_c"] = L.dept_id.astype("category").cat.codes
FEATURES = ["lag28", "lag35", "lag42", "lag49", "lag56", "rm7", "rm28", "rm56", "rwd4",
            "sell_price", "prel", "snap", "event", "wday", "month", "store_c", "dept_c"]
Y = L.pivot(index="id", columns="dn", values="y")
BASE = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2)

out = []
for T in ORIGINS:
    t0 = time.time()
    train = L[(L.dn <= T) & (L.dn > T - TRAIN_DAYS) & L.sell_price.notna() & L.rm56.notna()]
    test = L[(L.dn > T) & (L.dn <= T + HORIZON) & L.sell_price.notna()].copy()
    ds = lgb.Dataset(train[FEATURES], train.y, categorical_feature=["store_c", "dept_c"], free_raw_data=False)
    m_mean = lgb.train({**BASE, "objective": "tweedie", "tweedie_variance_power": 1.1}, ds, 400)
    m_q80 = lgb.train({**BASE, "objective": "quantile", "alpha": 0.8}, ds, 400)
    test["lgb_mean"] = np.clip(m_mean.predict(test[FEATURES]), 0, None)
    test["lgb_q80"] = np.clip(m_q80.predict(test[FEATURES]), 0, None)

    test["ma28"] = test.id.map(Y.loc[:, T - 27:T].mean(axis=1))
    test["wd8"] = np.nan
    test["wd8_q80"] = np.nan
    for d in range(T + 1, T + HORIZON + 1):
        same_weekday = [d - 7 * k for k in range(1, 20) if d - 7 * k <= T][:8]
        hist = Y.loc[:, same_weekday]
        idx = test.dn == d
        test.loc[idx, "wd8"] = test.loc[idx, "id"].map(hist.mean(axis=1)).values
        test.loc[idx, "wd8_q80"] = test.loc[idx, "id"].map(hist.quantile(0.8, axis=1)).values
    test["origin"] = T
    out.append(test[["id", "dept_id", "store_id", "dn", "date", "y", "sell_price",
                     "ma28", "wd8", "wd8_q80", "lgb_mean", "lgb_q80", "origin"]])
    print(f"window after day {T}: {time.time() - t0:.0f}s", flush=True)

pd.concat(out).to_parquet(WORK / "forecasts.parquet")
