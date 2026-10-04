# Better forecasts, same bad decisions

Code and results for the article *Better Forecasts, Same Bad Decisions* by Hardik Chawla.

The study asks one question: **does a more accurate forecast lead to cheaper inventory decisions?** On 4,311 Walmart food products, it did not. The most accurate forecast ranked fifth of six on decision cost. The approach with the least accurate forecast cost 17.1% less.

## What the study does

- **Data:** the public [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset of daily Walmart sales and prices (Makridakis, Spiliotis and Assimakopoulos, 2022). The study uses every food product in three stores (CA_3, TX_2, WI_2): 4,311 product-store pairs.
- **Test period:** three back-to-back four-week windows from February 29 to May 22, 2016, for 362,124 product-days. Each window is forecast using only data available before it starts.
- **Decision:** each forecast sets the stock level for one product on one day.
- **Costs:** a unit short costs 40% of its price; a unit left over costs 10% of its price. With this 4-to-1 ratio, the cheapest stock level covers demand on 8 of every 10 days (the 80th percentile).

## Six approaches compared

| # | Approach | Stocks to |
|---|---|---|
| 1 | Machine learning forecast (LightGBM, Tweedie loss) | Average |
| 2 | Average of the last 28 days | Average |
| 3 | Average of the last 8 same weekdays | Average |
| 4 | Machine learning forecast, converted to its 80th percentile (Poisson) | 80th percentile |
| 5 | 80th percentile of the last 8 same weekdays | 80th percentile |
| 6 | Machine learning model trained on the 80th percentile (pinball loss) | 80th percentile |

## Results

| Approach | Forecast error (units) | Decision cost (index) | Sales covered |
|---|---|---|---|
| Machine learning forecast | **1.60** (lowest) | 100.0 | 65.6% |
| 28-day average | 1.61 | 98.2 | 67.3% |
| Weekday average | 1.65 | 102.1 | 65.4% |
| Machine learning forecast, converted | 1.93 | 84.2 | 78.8% |
| Weekday rule, 80th percentile | 2.10 | 88.9 | 79.6% |
| Machine learning, trained on 80th percentile | 2.13 | **82.9** (lowest) | 82.6% |

Forecast error is mean absolute error per product-day. Decision cost is the total cost of shortages and leftovers, indexed to the machine learning forecast (100). Sales covered is the share of units sold that the plan had in stock.

Key findings:

1. The forecast accuracy ranking was close to the reverse of the decision cost ranking.
2. A spreadsheet rule (approach 5) cost 11.1% less than stocking to the machine learning forecast.
3. The saving from stocking at the right percentile grows with the cost ratio: 2.3% at 2:1, 15.8% at 4:1 and 37.5% at 9:1.
4. The quantile model saved between 16.2% and 18.0% in each of the three windows.

![Forecast error vs decision cost](figures/accuracy_vs_cost.png)

## Reproduce the results

Requires Python 3.10+ and about 8 GB of RAM. The forecasting step takes about 6 minutes on 2 CPU cores.

```bash
pip install -r requirements.txt
./get_data.sh                  # downloads the M5 files into ./data
python src/prepare_data.py     # builds the analysis panel
python src/forecast.py         # trains models and writes forecasts
python src/evaluate.py         # costs the decisions -> results/
python src/make_figures.py     # draws the figures -> figures/
```

## Limits

- M5 records sales, not demand. On days a shelf was empty, recorded sales are lower than true demand.
- Shortage and leftover costs are assumptions, not measured values.
- Each day is a separate decision with no carryover inventory. This fits fresh food better than shelf-stable goods.
- The LightGBM models are lightly tuned.

Treat the results as a measure of the gap between forecast accuracy and decision cost on one public dataset, not as an estimate for a specific retailer.

## Repository layout

```
src/paths.py          shared file locations
src/prepare_data.py   step 1: subset and reshape the M5 data
src/forecast.py       step 2: baselines and LightGBM forecasts
src/evaluate.py       step 3: stocking decisions, costs, sensitivity
src/make_figures.py   step 4: figures
results/              output tables
figures/              output charts
```

## References

- Ban, G.-Y., & Rudin, C. (2019). The big data newsvendor: Practical insights from machine learning. *Operations Research*, 67(1), 90-108.
- Elmachtoub, A.N., & Grigas, P. (2022). Smart "predict, then optimize." *Management Science*, 68(1), 9-26.
- Kourentzes, N., Trapero, J.R., & Barrow, D.K. (2020). Optimising forecasting models for inventory planning. *International Journal of Production Economics*, 225, 107597.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346-1364.

## License

Code: MIT License. The M5 data is not included; it is subject to the competition's terms.
