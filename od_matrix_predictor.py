"""
Minimal OD Matrix Predictor for Flow Prediction
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class ODMatrixPredictor:
    """Minimal OD matrix predictor using spatio-temporal features"""
    
    def __init__(self, n_stations: int, history_length: int = 6, feature_dim: int = 16):
        self.n_stations = n_stations
        self.H = history_length  # Historical snapshots
        self.d = feature_dim     # Feature dimension
        self.model = None
        
    def create_od_matrix(self, trips_df: pd.DataFrame, time_window: int = 60) -> np.ndarray:
        """Create OD matrix from trips data"""
        # Group trips by time windows
        trips_df['time_bin'] = trips_df['start_datetime'].dt.floor(f'{time_window}min')
        
        # Create OD matrix: [time_bins, n_stations, n_stations]
        time_bins = sorted(trips_df['time_bin'].unique())
        od_matrices = []
        
        for time_bin in time_bins:
            window_trips = trips_df[trips_df['time_bin'] == time_bin]
            od_matrix = np.zeros((self.n_stations, self.n_stations))
            
            # Fill OD matrix with trip counts
            for _, trip in window_trips.iterrows():
                origin_idx = hash(trip['start_station_id']) % self.n_stations
                dest_idx = hash(trip['end_station_id']) % self.n_stations
                od_matrix[origin_idx, dest_idx] += 1
                
            od_matrices.append(od_matrix)
            
        return np.array(od_matrices)
    
    def extract_features(self, stations_df: pd.DataFrame, od_matrices: np.ndarray, 
                        timestamp: datetime) -> np.ndarray:
        """Extract features for OD prediction"""
        T, N, _ = od_matrices.shape
        features = np.zeros((self.H, N, N, self.d))
        
        # Use last H time steps
        start_idx = max(0, T - self.H)
        
        for t in range(self.H):
            actual_t = start_idx + t
            if actual_t < T:
                od_matrix = od_matrices[actual_t]
                
                for i in range(N):
                    for j in range(N):
                        # Feature vector for OD pair (i,j)
                        feat_vec = np.zeros(self.d)
                        
                        # Past flow (most important)
                        feat_vec[0] = od_matrix[i, j]
                        
                        # Temporal features
                        feat_vec[1] = timestamp.hour / 24.0
                        feat_vec[2] = timestamp.weekday() / 7.0
                        feat_vec[3] = (timestamp.hour >= 7 and timestamp.hour <= 9) * 1.0  # Morning peak
                        feat_vec[4] = (timestamp.hour >= 17 and timestamp.hour <= 19) * 1.0  # Evening peak
                        
                        # Station features (if available)
                        if len(stations_df) > 0:
                            # Origin station features
                            feat_vec[5] = np.sum(od_matrix[i, :])  # Total outflow
                            feat_vec[6] = np.sum(od_matrix[:, i])  # Total inflow
                            
                            # Destination station features  
                            feat_vec[7] = np.sum(od_matrix[j, :])  # Total outflow
                            feat_vec[8] = np.sum(od_matrix[:, j])  # Total inflow
                        
                        # Distance proxy (simple)
                        feat_vec[9] = abs(i - j) / N  # Normalized station distance
                        
                        # Network centrality proxy
                        feat_vec[10] = (np.sum(od_matrix[i, :]) + np.sum(od_matrix[:, i])) / (2 * N)
                        feat_vec[11] = (np.sum(od_matrix[j, :]) + np.sum(od_matrix[:, j])) / (2 * N)
                        
                        features[t, i, j, :] = feat_vec
        
        return features
    
    def predict_next_od(self, features: np.ndarray) -> np.ndarray:
        """Predict next OD matrix"""
        H, N, _, d = features.shape
        
        # Simple baseline: weighted average of recent flows
        weights = np.exp(np.linspace(-2, 0, H))  # Recent timesteps get higher weight
        weights = weights / weights.sum()
        
        predicted_od = np.zeros((N, N))
        
        for t in range(H):
            # Extract flow features (first dimension)
            flow_matrix = features[t, :, :, 0]
            predicted_od += weights[t] * flow_matrix
        
        # Add temporal adjustment
        current_hour = features[-1, 0, 0, 1] * 24  # Extract hour
        
        # Peak hour multipliers
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            predicted_od *= 1.5  # Peak hours
        elif 22 <= current_hour or current_hour <= 6:
            predicted_od *= 0.3  # Night hours
        
        return predicted_od
    
    def get_top_flows(self, od_matrix: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Get top predicted flows for visualization"""
        flows = []
        
        for i in range(od_matrix.shape[0]):
            for j in range(od_matrix.shape[1]):
                if i != j and od_matrix[i, j] > 0:
                    flows.append({
                        'origin_idx': i,
                        'dest_idx': j,
                        'predicted_flow': od_matrix[i, j],
                        'confidence': min(1.0, od_matrix[i, j] / 10.0)  # Simple confidence
                    })
        
        # Sort by predicted flow and return top k
        flows.sort(key=lambda x: x['predicted_flow'], reverse=True)
        return flows[:top_k]

class ODMatrixVisualizer:
    """Minimal OD matrix visualization for maps"""
    
    @staticmethod
    def create_flow_lines(station_coords: Dict, top_flows: List[Dict], 
                         station_id_to_idx: Dict) -> List[Dict]:
        """Create flow lines for map visualization"""
        flow_lines = []
        
        # Reverse mapping: idx to station_id
        idx_to_station = {v: k for k, v in station_id_to_idx.items()}
        
        for flow in top_flows:
            origin_station = idx_to_station.get(flow['origin_idx'])
            dest_station = idx_to_station.get(flow['dest_idx'])
            
            if (origin_station in station_coords and 
                dest_station in station_coords):
                
                origin_coords = station_coords[origin_station]
                dest_coords = station_coords[dest_station]
                
                flow_lines.append({
                    'origin_coords': origin_coords,
                    'dest_coords': dest_coords,
                    'flow_value': flow['predicted_flow'],
                    'confidence': flow['confidence'],
                    'origin_station': origin_station,
                    'dest_station': dest_station
                })
        
        return flow_lines
    
    @staticmethod
    def get_flow_color(flow_value: float, max_flow: float) -> str:
        """Get color based on flow intensity"""
        intensity = min(1.0, flow_value / max_flow) if max_flow > 0 else 0
        
        if intensity > 0.7:
            return '#FF0000'  # Red for high flow
        elif intensity > 0.4:
            return '#FF8C00'  # Orange for medium flow
        else:
            return '#4169E1'  # Blue for low flow
    
    @staticmethod
    def get_line_width(flow_value: float, max_flow: float) -> float:
        """Get line width based on flow intensity"""
        intensity = min(1.0, flow_value / max_flow) if max_flow > 0 else 0
        return max(1.0, intensity * 8.0)  # Width between 1-8 pixels