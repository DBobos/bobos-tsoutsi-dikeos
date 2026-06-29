# Predicting Depression from Driving Data using Machine Learning

**Paper:** *Predicting depression from driving data using machine learning algorithms*  
**Authors:** D. Bobos, V. Tsoutsi

---

## Overview

This repository contains all Python scripts required to reproduce the analyses and figures presented in the paper. Participants drove a simulator at two sites (NTU and HMU) across three tracks (Motorway, Urban Low-Traffic, Rural). Driving behaviour features were extracted and used to classify participants as Controls or Patients (diagnosed with depression).

---

## Data

Raw data consists of per-participant `.xlsx` files exported from the driving simulator, organised as:

```
Simulator NTU/
  Controls/   ← CG* files
  Patients/   ← DP* files
Simulator HMU/
  Controls/
  Patients/
```

The scripts produce two main intermediate datasets that feed into the models:

| File | Description |
|---|---|
| `timeseries_data.csv` | Row-per-timestep data for all participants and tracks |
| `aggregated_data.csv` | One row per participant — mean/std features per track |
| `df_ntu_mtway.csv` | NTU motorway timeseries with route & bin columns |
| `df_ntu_urblt.csv` | NTU urban low-traffic timeseries with route & bin columns |

---

## Scripts — Run in this order

### Step 1 — Raw data ingestion
**`01_data_processing.py`**  
Reads all raw simulator `.xlsx` files, assigns participant class (Control/Patient) and location (NTU/HMU), extracts per-event mean values, and outputs a flat CSV per participant. This is the entry point; set the working directory to the folder containing the `Simulator NTU/` and `Simulator HMU/` subfolders.

---

### Step 2 — Build timeseries DataFrame
**`02_generate_timeseries_df.py`**  
Combines all per-participant CSVs from Step 1 into a single `timeseries_data.csv`. Also produces the track-specific DataFrames (`df_ntu_mtway.csv`, `df_ntu_urblt.csv`) with computed columns: `cumulative_distance`, `bins`, `route_x`, `route_y`, `lane_offset_m`.

---

### Step 3 — Displacement / lane-offset features
**`03_displacement_features.py`**  
Adds lateral displacement and lane-offset features to the timeseries DataFrames. Run after Step 2.

---

### Step 4 — Build aggregated DataFrame
**`04_generate_aggregated_df.py`**  
Reads the per-participant `.xlsx` files and computes summary statistics (mean, std, etc.) per participant per track, outputting `aggregated_data.csv`. This is the input for the Random Forest model in Step 5.

---

### Step 5 — Random Forest on aggregated features (Figure 5)
**`05_random_forest_aggregated.py`**  
Trains a Random Forest classifier (`n_estimators=200`, `class_weight='balanced'`, `max_depth=2`) on `aggregated_data.csv`. Outputs the horizontal bar-chart feature importance plot used as **Figure 5** in the paper. Contestants CG2, CG5, CG7 are excluded (crashed controls).

**Requires:** `aggregated_data.csv`

---

### Step 6 — Random Forest on raw timeseries (supplementary)
**`06_random_forest_timeseries.py`**  
Trains a Random Forest classifier on `timeseries_data.csv` (NTU location only) using SMOTEENN resampling. Reports accuracy and AUC; produces ROC curve and confusion matrix. Not a primary figure in the paper but supports the analysis.

**Requires:** `timeseries_data.csv`

---

### Step 7 — XGBoost on timeseries (Figure 6)
**`07_xgboost_timeseries.py`**  
Trains an XGBoost classifier on `timeseries_data.csv` (NTU only) with `RandomUnderSampler` balancing. Outputs the vertical bar-chart feature importance plot used as **Figure 6** in the paper.

**Requires:** `timeseries_data.csv`

---

### Step 8 — Motorway 3D map (Figure 1)
**`08_fig1_motorway_3d_map.py`**  
Generates the motorway 3D route map at 300 DPI. Uses the first clean segment of the motorway track (cumulative distance ≤ 2700 m) from `df_ntu_mtway.csv`, with teleport-row removal and Gaussian smoothing. Axes: X = pos_x, Y = pos_y, Z = normalised heading.

**Requires:** `df_ntu_mtway.csv`  
**Output:** `hires_plots/fig1_motorway_3d_map.png`

---

### Step 9 — Urban 3D map (Figure 3)
**`09_fig3_urban_3d_map.py`**  
Generates the urban low-traffic 3D route map at 300 DPI. Reconstructs the L-shaped circuit spatially (east arm + north arm) from `df_ntu_urblt.csv`. Axes: X = pos_x, Y = pos_y, Z = normalised heading.

**Requires:** `df_ntu_urblt.csv`  
**Output:** `hires_plots/fig3_urban_3d_map.png`

---

### Step 10 — All publication figures at 300 DPI
**`10_generate_all_publication_figures.py`**  
Convenience script that regenerates all six publication figures (Figs 1–6) at 300 DPI in one run. Steps 8 and 9 produce cleaner versions of Figs 1 and 3; use those if needed.

**Requires:** `aggregated_data.csv`, `timeseries_data.csv`, `df_ntu_mtway.csv`, `df_ntu_urblt.csv`  
**Output:** `hires_plots/fig1_motorway_3d_map.png` through `fig6_xgb_feature_importance.png`

---

## Dependencies

Install with:

```bash
pip install pandas numpy scipy matplotlib scikit-learn xgboost imbalanced-learn joblib openpyxl
```

Python 3.9+ recommended.

---

## Figure index

| Figure | Script | Description |
|---|---|---|
| Fig 1 | `08_fig1_motorway_3d_map.py` | Motorway 3D route map (pos_x, pos_y, heading) |
| Fig 2 | `10_generate_all_publication_figures.py` | Mean speed along motorway track by class |
| Fig 3 | `09_fig3_urban_3d_map.py` | Urban low-traffic 3D route map |
| Fig 4 | `10_generate_all_publication_figures.py` | Mean speed along urban track by class |
| Fig 5 | `05_random_forest_aggregated.py` | Random Forest feature importances (aggregated data) |
| Fig 6 | `07_xgboost_timeseries.py` | XGBoost feature importances (timeseries data) |
