# EPFL SURF 2025 — Enhanced Bike Flow Prediction
**Last updated:** August 2025

## Merged documentation source
This README consolidates the repository's individual Markdown documents:
- `CROSS_CITY_README.md` — cross-city transfer learning system, features and usage
- `CROSS_CITY_PREDICTION_REPORT.md` — experimental report and results
- `OSM_CACHE_README.md` — OSM caching system and CLI usage
- `LOCATION_BASED_OSM_GUIDE.md` — per-city location-based downloader guide
- `Internship_delivery.md` — enhanced system overview, feature lists and architecture

If you want the original long-form reports preserved, they remain in the repo — this README summarizes and links to them.

---

## Project summary

This repository implements a pipeline and research code for bike-sharing destination flow prediction using OSM-derived features, multi-path routing, and multiple ML approaches (statistical baselines, similarity-based models, and GNNs). It includes:

- OSM feature extraction and caching for ML
- Location-based and city-specific OSM downloaders
- Multi-path routing (OSRM / OpenRouteService) with multiple routing profiles
- Predictors: similarity-based, statistical baselines, and GNN experiments
- Streamlit interactive apps for exploration and click-to-predict workflows
- Evaluation and analysis code with comprehensive metrics and visualizations

## Quick navigation

- Getting started: Quick start & installation
- OSM cache: setup and CLI
- Location-based downloads per city
- Enhanced system: features and architecture
- Cross-city transfer: usage and report summary
- How to run apps and examples
- Troubleshooting, configuration, contribution, license

---

## Quick start

1. Clone and enter the repo:

```bash
git clone <repository-url>
cd EPFL_SURF_2025
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (recommended for PyG / advanced):

```bash
conda create -n surf2025 python=3.10 -y
conda activate surf2025
pip install -r requirements.txt
# Install PyTorch Geometric separately per your platform
```

3. Prepare OSM cache (one-time):

```bash
python setup_osm_cache.py           # interactive helper
python setup_osm_cache.py download  # non-interactive
```

4. Run the interactive apps:

```bash
streamlit run cross_city_main_app.py
streamlit run interactive_transfer_app.py
```

---

## OSM feature caching (from `OSM_CACHE_README.md`)

Purpose: pre-download OSM and population-derived features for stations to accelerate ML experiments.

What is downloaded
- Traditional OSM amenities (restaurants, hotels, shops, transit stops, parks, etc.)
- Population/demographic features
- Comprehensive OSMnx-based features at multiple radii (250m, 500m, 1000m)

Cache structure (example):

```
cache/
├── osm_features/
├── population_features/
├── comprehensive_osm/
├── consolidated/
└── ml_ready/
	├── station_features.csv
	└── all_features.json
```

Commands

```bash
python setup_osm_cache.py download
python setup_osm_cache.py status
python setup_osm_cache.py load
```

Advanced master downloader (when present):

```bash
python master_osm_cache_downloader.py --radius 500 --batch-size 30 --rate-limit 0.3
python master_osm_cache_downloader.py --force-refresh
```

Key options
- `--radius`: meters around station (default e.g., 1000)
- `--batch-size`: how many stations processed before saving
- `--rate-limit`: seconds between API requests to avoid throttle
- `--force-refresh`: redownload even if cached

Quick integration

```python
from setup_osm_cache import load_cached_features
features_df = load_cached_features()
X = features_df.drop(['station_id'], axis=1)
```

Troubleshooting hints
- If missing trips or cache files, run `python setup_osm_cache.py download`.
- For Overpass rate limits, increase `--rate-limit` or use smaller batch sizes.
- Install required packages: `pip install osmnx geopandas overpy`

---

## Location-based downloader (from `LOCATION_BASED_OSM_GUIDE.md`)

Use this to download per-city caches (faster than whole-repo downloads).

Examples

```bash
# Lausanne (small, fast)
python location_based_osm_downloader.py --city lausanne

# Bern (larger)
python location_based_osm_downloader.py --city bern --max-stations 50

# Custom bounds
python location_based_osm_downloader.py --bounds 46.4,6.4,46.6,6.8 --max-stations 15
```

Cache layout per city

```
cache/
├── lausanne/
│   ├── osm_features/
│   └── ml_ready/station_features.csv
├── bern/
└── zurich/
```

Tips
- Start development with Lausanne (65 stations) for faster iteration.
- Use `--max-stations` for quick tests.
- Combine city caches later if you need cross-city datasets.

---

## Enhanced system overview (from `Internship_delivery.md`)

Core capabilities
- OSM feature extraction: >40 types, multi-radius, density & accessibility metrics
- Multi-path routing: shortest, fastest, safest, scenic; supports OSRM and OpenRouteService
- ML evaluation: RMSE, MAE, R², MAPE, bootstrap confidence, CV and time-series validation
- Feature influence analysis: statistical and ML-based importance

Architecture (high level)

```
Enhanced Bike Flow Prediction System
├── OSM Feature Extractor
├── Multi-Path Router
├── ML Evaluation System
└── Enhanced Predictor
```

Routing profiles supported
- `cycling-regular`, `cycling-safe`, `cycling-fast`, `cycling-scenic`, `cycling-direct`

Examples (from docs)

```python
from osm_feature_extractor import OSMFeatureExtractor
extractor = OSMFeatureExtractor()
features = extractor.extract_features_around_station(lat, lon, radius_m=500)

from multi_path_router import MultiPathRouter
router = MultiPathRouter()
paths = router.get_multiple_paths(start_lat, start_lon, end_lat, end_lon, max_paths=5)
```

Configuration and env vars

```bash
export OPENROUTESERVICE_API_KEY="your_api_key"
export OSM_CACHE_DIR="cache/osm_features"
```

---

## Cross-city transfer system (from `CROSS_CITY_README.md`)

Purpose: predict spatial flows in a target city using models trained on a source city with domain adaptation and transfer learning.

Features
- GNN implementations: GCN, GAT, GraphSAGE
- Domain adaptation and feature alignment
- Interactive click-to-predict map UI
- Population-aware modeling and multi-feature engineering

High-level components

```
cross_city_transfer_predictor.py     # core transfer engine
enhanced_osm_integration.py         # OSM integration and router
cross_city_main_app.py              # Streamlit app
interactive_transfer_app.py         # alternative UI
```

Quick usage

```python
from cross_city_transfer_predictor import CrossCityTransferPredictor
predictor = CrossCityTransferPredictor(gnn_config, transfer_config)
predictor.load_source_city_data(trips_df, station_coords)
predictor.train_source_model()
predictor.prepare_target_city(target_stations, "zurich")
predictor.transfer_to_target_city()
flow, confidence = predictor.predict_target_flows('zur_001','zur_005', hour=17, day_of_week=1)
```

Interactive map workflow
- First click: source station (green)
- Second click: destination station (red)
- System calculates route, predicts flow, overlays visualization and confidence

Supported cities (current dataset)
- Bern, Zurich, Basel, Geneva, Lausanne (see `CROSS_CITY_README.md` for counts)

Configuration snippets

```python
GNNConfig(hidden_dim=128, num_layers=2, dropout=0.2, gnn_type='GCN', learning_rate=0.001)

TransferConfig(source_city='bern', target_city='zurich', fine_tune_epochs=50, domain_adaptation=True)
```

---

## Cross-city prediction report (from `CROSS_CITY_PREDICTION_REPORT.md`) — key takeaways

Experiment summary
- Dataset: ~91k trips over 8 days
- Tested methods: statistical averaging baseline, similarity-based 15-feature predictor, GNN attempts

Results
- Statistical averaging baseline: MAE 0.43, correlation 0.577 — promising
- Similarity-based approach: avg MAE 0.645, correlation 0.067 — limited transfer
- GNN: attempted but had stability/segfault issues on small data

Conclusions
- Direct transfer across cities (Lausanne→Bern) has limited effectiveness for advanced models; simple statistical transfer performs better as an initial approach.
- Recommended production approach: start with robust statistical baselines, collect local data for fine-tuning, and use GNNs primarily within cities where data is sufficient.

Artifacts generated in experiments
- Maps: `simple_cross_city_map.html`, `advanced_cross_city_map.html`
- Plots: `simple_cross_city_accuracy.png`, `advanced_cross_city_analysis.png`

---

## How to run core scripts (examples)

Prepare features and run a basic predictor

```bash
# Ensure cache exists
python setup_osm_cache.py download

# Run a baseline cross-city test (example script, adjust args)
python simple_cross_city_test.py --source lausanne --target bern
```

Run Streamlit apps

```bash
streamlit run cross_city_main_app.py
streamlit run interactive_transfer_app.py
```

Run GNN baselines (may require GPU/torch-geometric)

```bash
python run_gnn_baselines.py --config configs/gnn_config.yml
```

---

## Troubleshooting & tips

- Missing trips file: ensure `data/trips_8days_flat.csv` is present or update script args.
- API rate limits: increase `--rate-limit` and reduce `--batch-size` for the downloader.
- OSMnx/overpy missing: `pip install osmnx geopandas overpy`
- GNN segmentation faults: try smaller batches, correct PyTorch/Geometric install, or run on a machine with sufficient memory.

Performance suggestions
- Use city-limited caches for development. Start with Lausanne.
- Use consolidated `cache/ml_ready/station_features.csv` as your ML input to speed experiments.

---

## Tests & validation

There are analysis and evaluation scripts in the repo. Run targeted scripts to reproduce evaluation figures mentioned in the report. If you want, I can add a small smoke-test that loads `cache/ml_ready/station_features.csv` and runs a basic sanity check.

---

## Contributing

1. Fork the repository
2. Create a branch `git checkout -b feature/your-feature`
3. Commit and push
4. Open a PR with description and tests/examples

Development setup

```bash
git clone <repository-url>
pip install -r requirements_enhanced.txt  # for advanced features
```

---

## License & acknowledgments

This project is research work for EPFL SURF 2025. See LICENSE for details (MIT is referenced in docs). Acknowledgments: OpenStreetMap, OSRM, OpenRouteService, Streamlit, Scikit-learn, PyTorch, and other OSS projects.

---

If you'd like, I can now:
1. Add a short table of contents with internal anchors for quick navigation. 
2. Extract and append any sections from the original MD files that you want preserved verbatim (e.g., full report). 
3. Create a `docs/` folder and move full-length reports there while keeping a short summary in README.

Tell me which of these you'd like next and I'll apply it.
# EPFL_SURF_2025
Internship projects I did for EPFL in 2025
