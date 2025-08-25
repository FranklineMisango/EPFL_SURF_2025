import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')
STATIONS_CSV = os.path.join(DATA_DIR, 'unique_stations.csv')
POP_GEOJSON = os.path.join(DATA_DIR, 'pop_hectare_grid_latlon.geojson')

# Output files for each radius
OUTPUTS = {
    500: os.path.join(DATA_DIR, 'station_population_500m.csv'),
    1000: os.path.join(DATA_DIR, 'station_population_1000m.csv'),
    1500: os.path.join(DATA_DIR, 'station_population_1500m.csv'),
}

# Load stations
stations = pd.read_csv(STATIONS_CSV)
stations['geometry'] = stations.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

# TEST MODE: Only process the first 10 stations for speed
TEST_MODE = False
N_TEST = 10

stations_gdf = gpd.GeoDataFrame(stations, geometry='geometry', crs='EPSG:4326').to_crs(epsg=2056)  # Swiss projection
if TEST_MODE:
    print(f"[TEST MODE] Only processing the first {N_TEST} stations.")
    stations_gdf = stations_gdf.iloc[:N_TEST]

# Load population grid
pop_gdf = gpd.read_file(POP_GEOJSON)
if pop_gdf.crs is None or pop_gdf.crs.to_epsg() != 2056:
    pop_gdf = pop_gdf.to_crs(epsg=2056)

# Try to find the population column
pop_col = None
for c in pop_gdf.columns:
    if 'pop' in c.lower():
        pop_col = c
        break
if pop_col is None:
    raise ValueError('No population column found in population grid.')



import sys
import time
import numpy as np

def process_population_for_radius_optimized(radius, stations_gdf, pop_gdf, pop_col, output_path):
    results = []
    n_stations = len(stations_gdf)
    print(f"Processing radius {radius}m for {n_stations} stations...")
    start_time = time.time()

    # Build spatial index for population grid
    try:
        sindex = pop_gdf.sindex
        print("Spatial index built for population grid.")
    except Exception as e:
        print(f"Warning: Could not build spatial index: {e}")
        sindex = None

    for idx, station in stations_gdf.iterrows():
        t0 = time.time()
        buffer = station.geometry.buffer(radius)
        bbox = buffer.bounds
        # Pre-filter population cells by bounding box
        if sindex is not None:
            possible_matches_index = list(sindex.intersection(bbox))
            possible_matches = pop_gdf.iloc[possible_matches_index]
        else:
            possible_matches = pop_gdf[pop_gdf.geometry.bounds.apply(lambda b: (
                b[0] <= bbox[2] and b[2] >= bbox[0] and b[1] <= bbox[3] and b[3] >= bbox[1]), axis=1)]
        # Now do precise intersection
        pop_in_buffer = possible_matches[possible_matches.geometry.intersects(buffer)]
        total_pop = pop_in_buffer[pop_col].sum()
        results.append({
            'station_id': station['station_id'],
            'lat': station['lat'],
            'lon': station['lon'],
            f'population_{radius}m': total_pop
        })
        if (idx+1) % 10 == 0 or idx == n_stations-1:
            elapsed = time.time() - start_time
            print(f"  Processed {idx+1}/{n_stations} stations (last took {time.time()-t0:.2f}s, total elapsed: {elapsed:.1f}s)", flush=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f'Wrote {output_path} (radius {radius}m)')

for radius in [500, 1000, 1500]:
    print(f"\n--- Starting population extraction for {radius}m ---")
    process_population_for_radius_optimized(radius, stations_gdf, pop_gdf, pop_col, OUTPUTS[radius])
