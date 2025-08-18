import os
import sys
import pandas as pd
import pytest

# Ensure project root is on sys.path so tests can import local modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gnn_flow_predictor import GNNFlowPredictor


@pytest.mark.parametrize("radius", [500, 1000, 1500])
def test_explicit_cache_file_loads_correct_radius(radius):
    """When passed an explicit cache_file, the predictor should load that CSV.
    We assert the loaded dataframe contains the radius placeholder value (e.g., 500.0) which is used
    in the per-radius CSVs for missing/neighbourhood-capped distances.
    """
    cache_file = f"cache/ml_ready/switzerland_station_features_{radius}m.csv"
    assert os.path.exists(cache_file), f"Expected cache file missing: {cache_file}"

    # Construct with minimal trips_df (predictor will only load cached OSM features in __init__)
    predictor = GNNFlowPredictor(pd.DataFrame(), use_cached_features=True, cache_file=cache_file)

    assert predictor.cached_osm_df is not None, "cached_osm_df was not loaded"

    # Check that the radius placeholder value exists somewhere in the dataframe
    radius_value = float(radius)
    contains_radius = (predictor.cached_osm_df == radius_value).any().any()
    assert contains_radius, f"Loaded CSV {cache_file} does not contain radius placeholder {radius_value}"
