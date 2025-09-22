# PM2.5 Time Series Forecasting

Project: Beijing PM2.5 air-quality time series forecasting (educational / competition-style).

This repository contains a small project that prepares, trains, and evaluates LSTM/GRU time-series models for forecasting hourly PM2.5 concentrations. The work is organized as Jupyter notebooks and supporting folders for data, virtual environment, and output submissions.

Contents
--------
- `datasets/` - contains `train.csv` and `test.csv` used by the notebooks.
- `scripts/` - the main Jupyter notebooks used for EDA, preprocessing, model building, experiments, and submission generation. Important notebooks:
  - `Analysis_training_code.ipynb`  primary end-to-end notebook: EDA  preprocessing  generator creation  LSTM model build  experiments runner  produce per-experiment predictions and results CSVs.
  -
- `output_submissions/` - where experiment predictions, per-experiment plots, and `experiments_results.csv` (aggregated RMSEs) are written when you run the experiments loop.
- `env/` - a local Python virtual environment (optional; included for convenience).
- `requiremenst.txt` - a text file listing Python packages (note: filename contains a typo in this repo; see Installation).

Key features
------------
- Modular, documented notebook pipeline with separate cells for EDA, preprocessing, feature engineering, sequence (generator) creation, model build, training/evaluation, and submission generation.
- Memory-efficient sequence generator for training/validation to support larger datasets without loading all sequences into memory.
- Configurable LSTM builder and an experiments runner that programmatically runs multiple hyperparameter configurations and saves:
  - Per-experiment test predictions: `output_submissions/{exp_name}_predictions.csv`
  - Per-experiment preview plot: `output_submissions/{exp_name}_predictions_plot.png`
  - Aggregated results: `output_submissions/experiments_results.csv`

Quickstart
----------
1. Recommended: create and activate a Python virtual environment. On Windows (PowerShell):

```powershell
python -m venv env
& .\env\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requiremenst.txt
```

Note: the repository contains `requiremenst.txt` (typo). If you maintain your own list, ensure the environment has at least:

- Python 3.8+ (3.10/3.11 recommended)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyterlab or notebook
- tensorflow (2.x)

2. Open the main notebook in JupyterLab or Jupyter Notebook:

```powershell
jupyter lab
# or
jupyter notebook
```

3. Run cells from the top in order. Important order of sections to execute before running experiments:
- Data loading and datetime parsing cells
- `handle_missing_lstm` preprocessing cell
- `create_train_features` and `create_test_features` feature-engineering cells
- Encoding and scaling cells (they create `feature_scaler`, `target_scaler`, `train_featured_scaled`, `test_featured_scaled`, and `common_feature_cols`)
- `create_lstm_sequences_generator` (builds generators and `X_test` sequences)
- `build_lstm_model` cell
- `train_and_evaluate` cell (training helper)
- Experiments runner cell (this runs the grid of experiments and writes outputs)

Notes about running experiments
------------------------------
- Experiments can be long-running. The provided experiment grid in `Analysis_training.ipynb` contains 15 configurations with varying batch sizes, learning rates, layer counts, and epochs. Reduce `epochs` or the grid if you want a quick smoke test.
- The notebooks are written so model weights are not saved by default; instead, predictions (inverse-transformed to original pm2.5 scale) are saved per-experiment under `output_submissions/`.
- If you want to persist a model checkpoint, you can add a `ModelCheckpoint` callback in the `train_and_evaluate` function.

Outputs
-------
- `output_submissions/experiments_results.csv`  aggregated experiment metadata and RMSEs.
- `output_submissions/{exp_name}_predictions.csv`  per-experiment test predictions (one-column CSV named `prediction_pm2.5`).
- `output_submissions/{exp_name}_predictions_plot.png`  small plot of the first N predictions for visual inspection.
- Additional submission files (the project includes example `subm_fixed_*.csv` files)  these show the expected format for competition submissions.

Troubleshooting
---------------
- TypeErrors / mismatched function calls: If you modified function signatures, ensure the calls match keywords/parameters (the notebooks were recently updated to use keyword arguments for `train_and_evaluate`).
- Missing scalers / variables: If a cell errors because `target_scaler` or `common_feature_cols` are missing, re-run upstream cells (preprocessing/scaling) before running experiments.
- GPU / memory: Training LSTM models can be expedited with a GPU. If GPU is not available, reduce `batch_size`, `units`, or `epochs` to lower memory and runtime.
- Empty `X_test`: The sequence generator builds test sequences from `test.csv`. If you see errors around predictions, verify that `test.csv` has at least `lookback`+1 rows after preprocessing.

Recommended next steps
----------------------
- Run one short smoke experiment (set `epochs=3`) to validate end-to-end functionality.
- Add an experiments-summary cell to visualize `experiments_results.csv` (RMSE vs hyperparameters) and pick the best configuration for a longer run.
- Consider adding model checkpointing and saving the best model per-experiment if you want reproducible model files.

Thank you 
