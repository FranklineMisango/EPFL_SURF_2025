
# Flow Prediction with External Features (EPFL SURF 2025)
**Last updated:** August 2025

## Overview

This repository provides a comprehensive framework for spatio-temporal flow prediction using external features. It is designed for research and experimentation on predicting flows (e.g., people, vehicles, bikes, or goods) between locations, leveraging:

- Rich external features (OpenStreetMap, population, POIs, demographics, etc.)
- Multi-path routing and network-based features
- A wide range of machine learning models: tree ensembles, neural networks, GNNs, spatio-temporal models, and ensembles
- Robust evaluation, feature engineering, and reproducible experiments

The codebase is modular and supports both city-specific and cross-city transfer learning scenarios. While originally developed for bike-sharing flows, it is general and can be adapted to any flow prediction task with tabular, spatial, or temporal features.

## Key Features

- **External Feature Extraction:** Automated OSM and population feature extraction at multiple radii for each node/location.
- **Flexible ML Pipeline:** Supports XGBoost, LightGBM, CatBoost, Random Forest, MLP, TabNet, GNNs (GCN, GAT, GraphSAGE, GIN, Transformer), spatio-temporal models (DCRNN, ST-GCN, TFT), and ensemble methods (stacking, blending).
- **Multi-Path Routing:** Integrates OSRM/OpenRouteService for shortest, fastest, safest, and scenic path features.
- **Advanced Evaluation:** RMSE, MAE, R², MAPE, cross-validation, time-based splits, and feature importance analysis.
- **Interactive Apps:** Streamlit UIs for interactive prediction, feature exploration, and transfer learning demos.
- **Cross-City Transfer:** Tools for domain adaptation and transfer learning between cities or regions.

## Example Use Cases

- Predicting demand or flows for bike/scooter sharing, ride-hailing, logistics, or public transport
- Modeling flows between any spatial nodes (stations, stops, warehouses, etc.)
- Benchmarking ML, GNN, and ensemble models on real-world spatio-temporal data
- Studying the impact of external features (POIs, population, OSM) on flows

## Quick Start

```bash
git clone <repository-url>
cd EPFL_SURF_2025
pip install -r requirements.txt
# (Optional) conda create -n surf2025 python=3.10 -y && conda activate surf2025
# Install PyTorch Geometric and other extras as needed
```

Prepare OSM/population features:

```bash
python setup_osm_cache.py download
```

Run a baseline or advanced model:

```bash
python helpers/run_gnn_baselines.py
```

Launch the interactive app:

```bash
streamlit run cross_city_main_app.py
```

## Model Support

The framework supports the following model families:

- **Tree Ensembles:** XGBoost, LightGBM, CatBoost, Random Forest
- **Neural Networks:** MLP, TabNet
- **Graph Neural Networks:** GCN, GAT, GraphSAGE, GIN, Transformer
- **Spatio-Temporal Models:** DCRNN, ST-GCN, Temporal Fusion Transformer (TFT)
- **Ensembles:** Stacking, Blending (combine any of the above)

All models can use the same feature pipeline, and results are logged and summarized for easy comparison.

## Data & Features

- **Input:** Trip/flow records (CSV), OSM features, population/demographics, routing features
- **Feature Engineering:** Automated, multi-radius, supports custom features
- **Cache:** All features are cached for reproducibility and fast iteration

## Evaluation & Results

After training, the script logs RMSE, MAE, and R² for each model and radius. Results can be printed, saved, or visualized. The framework supports robust validation and comparison across models and feature sets.

## Extending & Customizing

- Add new features by editing the feature extraction scripts
- Add new models by implementing a new baseline in `helpers/run_gnn_baselines.py`
- Use your own data by placing it in the `data/` folder and updating the config/scripts

## Documentation

See the original markdown files in the repo for detailed guides on OSM caching, location-based downloads, cross-city transfer, and experimental reports. This README provides a technical summary; for full details, consult the respective docs.

## License

MIT License (see LICENSE file). Developed as part of EPFL SURF 2025 research.


