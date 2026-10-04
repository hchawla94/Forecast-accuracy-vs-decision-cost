"""Step 1: build the analysis panel from the raw M5 files.

Keeps every FOODS product in one store per state (CA_3, TX_2, WI_2),
the last 900 days of daily sales, plus calendar and price fields.
"""
import numpy as np
import pandas as pd
from paths import DATA, WORK

STORES = ["CA_3", "TX_2", "WI_2"]
LAST_DAY = 1941
N_DAYS = 900

sales = pd.read_csv(DATA / "sales_train_evaluation.csv")
sales = sales[(sales.cat_id == "FOODS") & (sales.store_id.isin(STORES))]
sales["id"] = sales.item_id + "_" + sales.store_id
day_cols = [f"d_{i}" for i in range(LAST_DAY - N_DAYS + 1, LAST_DAY + 1)]
keys = ["id", "item_id", "dept_id", "store_id"]
panel = sales[keys + day_cols].melt(id_vars=keys, var_name="d", value_name="y")
panel["dn"] = panel.d.str[2:].astype(int)

cal = pd.read_csv(DATA / "calendar.csv")
cal["dn"] = np.arange(1, len(cal) + 1)
panel = panel.merge(
    cal[["dn", "wm_yr_wk", "wday", "month", "event_name_1", "snap_CA", "snap_TX", "snap_WI", "date"]],
    on="dn",
)

prices = pd.read_csv(DATA / "sell_prices.csv")
prices = prices[prices.store_id.isin(STORES) & prices.item_id.str.startswith("FOODS")]
panel = panel.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

state = panel.store_id.str[:2]
panel["snap"] = np.select([state == "CA", state == "TX"], [panel.snap_CA, panel.snap_TX], panel.snap_WI)
panel["event"] = panel.event_name_1.notna().astype(int)
panel = panel.drop(columns=["snap_CA", "snap_TX", "snap_WI", "event_name_1", "d"])
panel.to_parquet(WORK / "foods_panel.parquet")
print(f"{panel.id.nunique()} series, {len(panel):,} rows")
