# OSM Feature Caching for ML Algorithms

## Quick Start

Before running your ML algorithms, you need to download and cache OSM features. This is a **one-time setup** that creates reusable cached data.

### 1. Simple Interactive Setup

```bash
python setup_osm_cache.py
```

This will guide you through the process interactively.

### 2. Command Line Usage

```bash
# Download all OSM features
python setup_osm_cache.py download

# Check cache status
python setup_osm_cache.py status

# Load cached features (for testing)
python setup_osm_cache.py load
```

### 3. Advanced Usage

```bash
# Full control with master downloader
python master_osm_cache_downloader.py --help

# Quick download with default settings
python master_osm_cache_downloader.py

# Custom radius and faster processing
python master_osm_cache_downloader.py --radius 500 --batch-size 30 --rate-limit 0.3

# Force refresh all caches
python master_osm_cache_downloader.py --force-refresh

# Check current status
python master_osm_cache_downloader.py --status
```

## What Gets Downloaded

The system downloads three types of OSM features around each bike station:

### 1. Traditional OSM Features (`trad_*`)
- **Hotels, restaurants, cafes** - Tourism and dining destinations
- **Banks, ATMs, shops** - Commercial services
- **Schools, universities, libraries** - Educational institutions  
- **Hospitals, clinics, pharmacies** - Healthcare facilities
- **Parks, sports centers** - Recreation areas
- **Bus stops, train stations** - Transportation hubs
- **Land use zones** - Residential, commercial, industrial areas

### 2. Population Features (`pop_*`)
- **Population density** - How many people live/work nearby
- **Demographic data** - Age, employment statistics
- **Housing density** - Residential building concentration
- **Commercial activity** - Business and retail density

### 3. Comprehensive OSM Features (`comp_*`)
- **Multi-radius analysis** - Features within 250m, 500m, 1000m
- **Feature diversity scores** - How varied the nearby amenities are
- **Density calculations** - Features per km²
- **Spatial relationships** - Distance to nearest amenities

## Cache Structure

After running the setup, you'll have this cache structure:

```
cache/
├── osm_features/           # Individual traditional OSM cache files
├── population_features/    # Individual population cache files  
├── comprehensive_osm/      # Individual comprehensive cache files
├── consolidated/           # Consolidated caches by type
│   ├── traditional_osm_features.json
│   ├── population_features.json
│   └── comprehensive_osm_features.json
├── ml_ready/              # 🎯 ML-ready unified features
│   ├── station_features.csv     # Your main file for ML!
│   ├── all_features.json        # Same data in JSON format
│   └── feature_metadata.json    # Feature descriptions
└── download_summary.json  # Download statistics and logs
```

## Using Cached Features in Your ML Code

### Load Features for ML

```python
import pandas as pd

# Load the cached features (this is fast!)
features_df = pd.read_csv('cache/ml_ready/station_features.csv')

print(f"Loaded {len(features_df)} stations with {len(features_df.columns)-1} features")

# Your features are ready for ML algorithms
X = features_df.drop(['station_id'], axis=1)  # Remove non-feature columns
```

### Example Integration

```python
# Before running your ML algorithm
from setup_osm_cache import load_cached_features

# Load cached OSM features
osm_features_df = load_cached_features()

if osm_features_df is None:
    print("OSM features not cached. Run setup first:")
    print("python setup_osm_cache.py download")
    exit(1)

# Now combine with your trip data
trips_df = pd.read_csv('data/trips_8days_flat.csv')

# Merge trip destinations with OSM features
trips_with_features = trips_df.merge(
    osm_features_df, 
    left_on='end_station_id', 
    right_on='station_id', 
    how='left'
)

# Run your ML algorithm with enriched data
from your_ml_algorithm import train_model
model = train_model(trips_with_features)
```

## Configuration Options

### Master Downloader Options

| Option | Default | Description |
|--------|---------|-------------|
| `--trips-file` | `data/trips_8days_flat.csv` | Input trips data file |
| `--cache-dir` | `cache` | Base directory for all caches |
| `--radius` | `1000` | Radius in meters around each station |
| `--batch-size` | `20` | Stations to process before saving |
| `--rate-limit` | `0.5` | Seconds between API requests |
| `--force-refresh` | `False` | Ignore existing caches |
| `--no-traditional` | `False` | Skip traditional OSM features |
| `--no-osmnx` | `False` | Skip comprehensive OSM features |
| `--no-population` | `False` | Skip population features |

### Performance Tuning

```bash
# Faster download (higher API load)
python master_osm_cache_downloader.py --rate-limit 0.2 --batch-size 50

# Conservative download (lower API load)  
python master_osm_cache_downloader.py --rate-limit 1.0 --batch-size 10

# Smaller radius for faster processing
python master_osm_cache_downloader.py --radius 500
```

## Troubleshooting

### Common Issues

1. **Missing trips file**
   ```
   ❌ Trips file not found: data/trips_8days_flat.csv
   ```
   **Solution:** Make sure your trips CSV file is in the `data/` directory.

2. **API rate limits**
   ```
   ❌ Overpass API error: rate limit exceeded
   ```
   **Solution:** Increase `--rate-limit` to 1.0 or higher.

3. **Missing dependencies**
   ```
   ❌ Failed to initialize OSM extractor: No module named 'osmnx'
   ```
   **Solution:** Install missing packages:
   ```bash
   pip install osmnx geopandas overpy
   ```

4. **Partial cache exists**
   ```
   📦 Loading existing traditional OSM cache...
   ```
   **Solution:** Use `--force-refresh` to redownload everything.

### Recovery from Failed Downloads

```bash
# Check what's already cached
python master_osm_cache_downloader.py --status

# Resume/complete partial download
python master_osm_cache_downloader.py

# Start fresh if needed
python master_osm_cache_downloader.py --force-refresh
```

### Cache Cleanup

```bash
# Remove all caches and start fresh
rm -rf cache/
python setup_osm_cache.py download

# Clean up individual files (keep only consolidated)
python master_osm_cache_downloader.py --cleanup
```

## Integration with Existing Code

### Update Your ML Pipeline

1. **Add cache check at start:**
   ```python
   from setup_osm_cache import quick_cache_download, load_cached_features
   
   # Ensure features are cached
   osm_features = load_cached_features()
   if osm_features is None:
       print("Downloading OSM features...")
       quick_cache_download()
       osm_features = load_cached_features()
   ```

2. **Use in your GNN model:**
   ```python
   # In gnn_flow_predictor.py
   def prepare_node_features(self, station_coords):
       # Load cached OSM features
       osm_features = load_cached_features()
       
       # Merge with station coordinates
       # ... existing code ...
   ```

3. **Use in multi-path router:**
   ```python
   # In multi_path_router.py  
   def calculate_route_attractiveness(self, route):
       # Load cached OSM features for destination
       osm_features = load_cached_features()
       destination_features = osm_features[osm_features.station_id == route.destination]
       # ... calculate attractiveness based on features ...
   ```

## Feature Descriptions

### Key Feature Categories

- **`trad_restaurants_count`** - Number of restaurants within radius
- **`trad_hotels_density`** - Hotel density (per km²)
- **`pop_population_total`** - Total population in area
- **`comp_osmnx_amenities_count_500m`** - Amenities within 500m
- **`comp_total_all_features`** - Total feature count from all sources

### Feature Naming Convention

- **`trad_*`** - Traditional OSM features from Overpass API
- **`pop_*`** - Population and demographic features  
- **`comp_*`** - Comprehensive features from OSMnx
- **`lat`, `lon`** - Station coordinates
- **`station_id`** - Unique station identifier

## Performance Expectations

### Download Time Estimates

| Stations | Traditional | Population | Comprehensive | Total |
|----------|-------------|------------|---------------|-------|
| 100 | ~5 min | ~2 min | ~10 min | ~17 min |
| 500 | ~25 min | ~10 min | ~45 min | ~80 min |
| 1000 | ~50 min | ~20 min | ~90 min | ~160 min |

### Cache Sizes

| Feature Type | Size per 1000 stations |
|--------------|-------------------------|
| Traditional OSM | ~5-10 MB |
| Population | ~2-5 MB |
| Comprehensive | ~10-20 MB |
| **Total** | **~17-35 MB** |

## Next Steps

1. **Run the cache setup** - This downloads all OSM features
2. **Verify the cache** - Check that `cache/ml_ready/station_features.csv` exists
3. **Update your ML code** - Load cached features instead of downloading each time
4. **Run your algorithms** - Enjoy faster execution with pre-cached data!

The cached features will significantly improve your ML algorithm performance and provide rich contextual information about each bike station's surroundings.
