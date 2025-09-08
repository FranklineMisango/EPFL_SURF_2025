#!/usr/bin/env python3
"""
Generate OD matrix JSON file for visualization
"""

import pandas as pd
import numpy as np
import json

def generate_od_matrix():
    """Generate a sample OD matrix from the existing data"""
    
    # Load stations
    try:
        stations_df = pd.read_json('stations.json')
        print(f"Loaded {len(stations_df)} stations")
    except Exception as e:
        print(f"Error loading stations: {e}")
        return
    
    # Load predicted flows
    try:
        flows_df = pd.read_json('predicted_flows.json')
        print(f"Loaded {len(flows_df)} flow predictions")
    except Exception as e:
        print(f"Error loading flows: {e}")
        return
    
    # Get unique station IDs
    station_ids = sorted(stations_df['station_id'].unique())
    n_stations = len(station_ids)
    
    # Create station ID to index mapping
    station_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}
    
    # Initialize OD matrix
    od_matrix = np.zeros((n_stations, n_stations))
    
    # Fill OD matrix with predicted flows
    for _, flow in flows_df.iterrows():
        origin_id = flow['origin']
        dest_id = flow['destination']
        predicted_flow = flow['predicted_flow']
        
        if origin_id in station_to_idx and dest_id in station_to_idx:
            origin_idx = station_to_idx[origin_id]
            dest_idx = station_to_idx[dest_id]
            od_matrix[origin_idx, dest_idx] = predicted_flow
    
    # Convert to DataFrame with station IDs as labels
    od_df = pd.DataFrame(od_matrix, index=station_ids, columns=station_ids)
    
    # Export to JSON in the format expected by the HTML table
    od_json = od_df.to_json(orient='split')
    
    # Save to file
    with open('od_matrix.json', 'w') as f:
        f.write(od_json)
    
    print(f"Generated od_matrix.json with {n_stations}x{n_stations} matrix")
    print(f"Total non-zero flows: {np.count_nonzero(od_matrix)}")
    print(f"Max flow value: {od_matrix.max():.4f}")
    print(f"Average flow value: {od_matrix.mean():.4f}")

if __name__ == "__main__":
    generate_od_matrix()