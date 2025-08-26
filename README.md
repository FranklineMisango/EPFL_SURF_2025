
# DCRNN Flow Prediction Framework (EPFL SURF 2025)
**Last updated:** January 2025

## Overview

This repository implements a Diffusion Convolutional Recurrent Neural Network (DCRNN) framework for spatio-temporal flow prediction in bike-sharing systems. The framework combines graph neural networks with recurrent architectures to capture both spatial dependencies between stations and temporal patterns in flow data.

**Core Features:**
- Full DCRNN implementation with diffusion convolution and GRU cells
- Chunked processing for scalability to large station networks
- Rich external features from OpenStreetMap, population data, and POIs
- Interactive web-based visualization dashboard
- Comprehensive evaluation metrics and model diagnostics

The system processes historical flow data and station features to predict future bike flows between stations, making it valuable for demand forecasting and system optimization.

## Key Features

**DCRNN Architecture:**
- Diffusion Convolutional layers for spatial message passing
- GRU cells for temporal sequence modeling
- Adjacency matrix based on station proximity
- Chunked processing for memory efficiency

**Data Processing:**
- External feature integration (OSM, population, POIs)
- Multi-radius feature extraction around stations
- Automated OD matrix generation from trip data
- Temporal windowing with configurable history length

**Visualization Dashboard:**
- Interactive web-based flow visualization
- Real-time filtering and threshold adjustment
- Station markers sized by flow volume
- Flow lines colored by prediction magnitude
- Live performance metrics display

**Evaluation Framework:**
- RMSE, MAE, and R² metrics
- Model performance tracking
- Prediction quality assessment
- Comprehensive logging and diagnostics

## Use Cases

- **Bike-sharing demand forecasting:** Predict hourly flows between stations
- **System optimization:** Identify high-demand routes and station imbalances
- **Capacity planning:** Forecast future demand for infrastructure decisions
- **Real-time operations:** Support dynamic bike redistribution strategies
- **Research applications:** Benchmark spatio-temporal prediction methods

## Quick Start

### Installation

```bash
git clone <repository-url>
cd EPFL_SURF_2025
pip install -r requirements.txt
```

### Run DCRNN Training

```bash
# Train DCRNN model and generate predictions
python train_predict_od_dcrnn.py

# Or use the full DCRNN runner (ensures no minimal fallbacks)
python run_full_dcrnn_only.py
```

### Launch Visualization Dashboard

```bash
# Start the enhanced visualization server
python serve_enhanced_viz.py
```

This opens an interactive dashboard at `http://localhost:8001/enhanced_flow_viz.html` showing:
- Predicted flow patterns between stations
- Model performance metrics
- Interactive filtering and exploration tools
- Real-time statistics

### Generate OD Matrix

```bash
# Create origin-destination matrix for analysis
python generate_od_matrix.py
```

## DCRNN Architecture

**Core Components:**
- **Diffusion Convolution:** Captures spatial dependencies using graph structure
- **GRU Cells:** Model temporal sequences and maintain memory
- **Adjacency Matrix:** Defines station connectivity based on geographic proximity
- **Feature Integration:** Combines temporal flows with static station features

**Model Configuration:**
- Hidden dimensions: 64
- Sequence length: 6 timesteps (6 hours history)
- Prediction horizon: 1 timestep (1 hour ahead)
- Batch processing with chunked station handling
- Adam optimizer with configurable learning rate

## Data Pipeline

**Input Data:**
- `trips_8days_flat.csv`: Historical trip records with timestamps
- `switzerland_station_features_1000m_with_pop.csv`: Station features (OSM, population)
- `unique_stations.csv`: Station coordinates and metadata

**Feature Engineering:**
- Temporal OD matrices generated from trip data
- Station features: OSM amenities, population density, POI counts
- Multi-radius feature extraction (500m, 1000m, 1500m)
- Gravity-based scoring combining distance and feature density

**Output:**
- `predicted_flows.json`: Flow predictions for visualization
- `model_metrics.json`: Performance metrics and diagnostics
- `od_matrix.json`: Origin-destination flow matrices

## Evaluation Metrics

**Model Performance:**
- **RMSE:** Root Mean Square Error for prediction accuracy
- **MAE:** Mean Absolute Error for average prediction deviation
- **R²:** Coefficient of determination for variance explained

**Visualization Metrics:**
- Flow threshold filtering for noise reduction
- Top-K flow identification for pattern analysis
- Station activity levels and flow imbalances
- Real-time prediction statistics in dashboard

## Configuration

**Model Parameters:**
Edit `full_dcrnn_config.py` to modify:
- Network architecture (hidden dimensions, layers)
- Training parameters (learning rate, epochs, batch size)
- Data processing (sequence length, prediction horizon)
- Station limits and chunking for memory management

**Data Sources:**
Replace input files in `Data/` folder:
- Trip records: Update `trips_file` path in config
- Station features: Update `station_features_file` path
- Ensure consistent station IDs across all data files

## File Structure

```
EPFL_SURF_2025/
├── train_predict_od_dcrnn.py    # Main DCRNN training script
├── full_dcrnn_config.py         # Model configuration
├── run_full_dcrnn_only.py       # Full DCRNN runner
├── serve_enhanced_viz.py        # Visualization server
├── enhanced_flow_viz.html       # Interactive dashboard
├── generate_od_matrix.py        # OD matrix generation
├── Data/                        # Input datasets
│   ├── trips_8days_flat.csv
│   ├── switzerland_station_features_*.csv
│   └── unique_stations.csv
├── helpers/                     # Utility scripts
└── results/                     # Output directory
```

## License

MIT License (see LICENSE file). Developed as part of EPFL SURF 2025 research.


