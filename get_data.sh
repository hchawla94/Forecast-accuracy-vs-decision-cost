#!/usr/bin/env bash
# Download the M5 competition files into ./data.
# Official source: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data (Kaggle login required).
# The command below uses Nixtla's public mirror of the same files.
set -euo pipefail
mkdir -p data
curl -L -o data/m5.zip https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip
unzip -o data/m5.zip -d data
