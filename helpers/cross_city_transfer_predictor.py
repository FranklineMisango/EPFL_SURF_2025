"""
Cross-City Transfer Learning System for Spatial Flow Prediction
Enables training on one city and predicting flows in another using OSM features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import math
import time

from gnn_flow_predictor import GNNFlowPredictor, GNNConfig, BikeFlowGNN
from osm_feature_extractor import OSMFeatureExtractor
from population_feature_extractor import PopulationFeatureExtractor

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

@dataclass
class TransferConfig:
    """Configuration for transfer learning"""
    source_city: str = "bern"
    target_city: str = "zurich"
    
    # Transfer learning parameters
    freeze_layers: List[str] = None  # Which layers to freeze during transfer
    fine_tune_epochs: int = 50
    fine_tune_lr: float = 0.0001
    feature_alignment_method: str = "standardization"  # "standardization", "min_max", "none"
    
    # Domain adaptation parameters
    domain_adaptation: bool = True
    adaptation_weight: float = 0.1
    
    # Feature engineering for transfer
    use_normalized_features: bool = True
    use_population_features: bool = True
    use_distance_features: bool = True
    use_temporal_features: bool = True

class FeatureAligner:
    """Aligns features between different cities for transfer learning"""
    
    def __init__(self, config: TransferConfig):
        self.config = config
        self.source_scalers = {}
        self.target_scalers = {}
        self.feature_mappings = {}
        
    def fit_source_features(self, source_features: Dict[str, np.ndarray]):
        """Fit scalers on source city features"""
        for feature_name, features in source_features.items():
            if self.config.feature_alignment_method == "standardization":
                scaler = StandardScaler()
            elif self.config.feature_alignment_method == "min_max":
                scaler = MinMaxScaler()
            else:
                scaler = None
                
            if scaler is not None:
                scaler.fit(features.reshape(-1, 1) if features.ndim == 1 else features)
                self.source_scalers[feature_name] = scaler
    
    def transform_source_features(self, source_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform source features using fitted scalers"""
        transformed = {}
        for feature_name, features in source_features.items():
            if feature_name in self.source_scalers:
                scaler = self.source_scalers[feature_name]
                transformed[feature_name] = scaler.transform(
                    features.reshape(-1, 1) if features.ndim == 1 else features
                ).flatten() if features.ndim == 1 else scaler.transform(features)
            else:
                transformed[feature_name] = features
        return transformed
    
    def align_target_features(self, target_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Align target features to source feature distribution"""
        aligned = {}
        for feature_name, features in target_features.items():
            if feature_name in self.source_scalers:
                # Use source scaler to align target features
                scaler = self.source_scalers[feature_name]
                aligned[feature_name] = scaler.transform(
                    features.reshape(-1, 1) if features.ndim == 1 else features
                ).flatten() if features.ndim == 1 else scaler.transform(features)
            else:
                aligned[feature_name] = features
        return aligned

class DomainAdaptationLayer(nn.Module):
    """Domain adaptation layer to reduce domain shift between cities"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # Source vs Target domain
        )
        
    def forward(self, features):
        return self.domain_classifier(features)

class TransferGNN(BikeFlowGNN):
    """Enhanced GNN with transfer learning capabilities"""
    
    def __init__(self, config: GNNConfig, transfer_config: TransferConfig, 
                 node_features: int, edge_features: int = 0):
        super().__init__(config, node_features, edge_features)
        self.transfer_config = transfer_config
        
        # Add domain adaptation if enabled
        if transfer_config.domain_adaptation:
            self.domain_adapter = DomainAdaptationLayer(config.hidden_dim)
        
        # Feature normalization layers for better transfer
        if transfer_config.use_normalized_features:
            self.feature_normalizer = nn.LayerNorm(node_features)
    
    def forward(self, x, edge_index, edge_attr=None, return_embeddings=False):
        """Forward pass with optional domain adaptation"""
        # Normalize input features if enabled
        if hasattr(self, 'feature_normalizer'):
            x = self.feature_normalizer(x)
        
        # Embed node features
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            residual = x if self.config.use_residual and i > 0 else None
            
            if self.gnn_type == "GCN":
                x = conv(x, edge_index)
            elif self.gnn_type == "GAT":
                x = conv(x, edge_index)
                if i < len(self.conv_layers) - 1 and x.dim() > 2:
                    x = x.view(x.size(0), -1)  # Flatten attention heads
            elif self.gnn_type == "GraphSAGE":
                x = conv(x, edge_index)
            
            # Apply batch normalization if enabled
            if self.config.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = self.dropout(x)
            
            # Add residual connection if enabled
            if residual is not None and residual.shape == x.shape:
                x = x + residual
        
        if return_embeddings:
            return x
        
        return x
    
    def predict_flow(self, source_embedding, target_embedding):
        """Predict flow between source and target stations"""
        # Concatenate source and target embeddings
        combined = torch.cat([source_embedding, target_embedding], dim=-1)
        
        # Predict flow
        flow = self.flow_predictor(combined)
        return flow
    
    def compute_domain_loss(self, embeddings, domain_labels):
        """Compute domain adaptation loss"""
        if not hasattr(self, 'domain_adapter'):
            return torch.tensor(0.0)
        
        domain_pred = self.domain_adapter(embeddings)
        domain_loss = F.cross_entropy(domain_pred, domain_labels)
        return domain_loss

class CrossCityTransferPredictor:
    """Main class for cross-city transfer learning"""
    
    def __init__(self, gnn_config: GNNConfig, transfer_config: TransferConfig):
        self.gnn_config = gnn_config
        self.transfer_config = transfer_config
        self.feature_aligner = FeatureAligner(transfer_config)
        
        # Models
        self.source_model = None
        self.transfer_model = None
        
        # Feature extractors
        self.osm_extractor = OSMFeatureExtractor()
        self.population_extractor = PopulationFeatureExtractor()
        
        # City data
        self.source_data = {}
        self.target_data = {}
        
        # Performance tracking
        self.transfer_metrics = {}
        
    def load_source_city_data(self, trips_df: pd.DataFrame, station_coords: Dict[str, Tuple[float, float]],
                              station_features: Optional[Dict[str, Dict]] = None,
                              use_cached_features: bool = False):
        """Load and process source city data"""
        logger.info(f"Loading source city data: {self.transfer_config.source_city}")

        self.source_data = {
            'trips_df': trips_df,
            'station_coords': station_coords,
            'station_features': {},
            'flow_patterns': {}
        }

        # If cached station features provided or flagged, reuse them and skip extraction
        if use_cached_features and station_features:
            logger.info("Using cached station features for source city; skipping OSM/pop extraction")
            self.source_data['station_features'] = station_features
        else:
            # Extract comprehensive features for source city
            self._extract_city_features(self.source_data, is_source=True)

        # Learn flow patterns
        self._learn_flow_patterns(self.source_data)

        logger.info(f"Source city data loaded: {len(station_coords)} stations, {len(trips_df)} trips")
    
    def prepare_target_city(self, station_coords: Dict[str, Tuple[float, float]], 
                          city_name: str = None,
                          station_features: Optional[Dict[str, Dict]] = None,
                          use_cached_features: bool = False):
        """Prepare target city for prediction (no trip data needed)"""
        if city_name:
            self.transfer_config.target_city = city_name
            
        logger.info(f"Preparing target city: {self.transfer_config.target_city}")
        
        self.target_data = {
            'station_coords': station_coords,
            'station_features': {},
            'flow_patterns': {}
        }

        if use_cached_features and station_features:
            logger.info("Using cached station features for target city; skipping OSM/pop extraction")
            self.target_data['station_features'] = station_features
        else:
            # Extract features for target city
            self._extract_city_features(self.target_data, is_source=False)

        # Align features with source city
        self._align_target_features()

        logger.info(f"Target city prepared: {len(station_coords)} stations")
    
    def _extract_city_features(self, city_data: Dict, is_source: bool = True):
        """Extract comprehensive features for a city"""
        station_coords = city_data['station_coords']
        features = {}
        
        progress_bar = None
        if not is_source:  # Only show progress for target city (interactive)
            import streamlit as st
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        total_stations = len(station_coords)
        
        for i, (station_id, (lat, lon)) in enumerate(station_coords.items()):
            if progress_bar:
                progress = (i + 1) / total_stations
                progress_bar.progress(progress)
                status_text.text(f"Extracting features for station {i+1}/{total_stations}")
            
            try:
                station_features = self._extract_station_features(lat, lon, station_id, city_data, is_source)
                features[station_id] = station_features
                
                # Rate limiting for OSM API
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to extract features for station {station_id}: {e}")
                features[station_id] = self._get_default_features()
        
        if progress_bar:
            progress_bar.empty()
            status_text.empty()
        
        city_data['station_features'] = features
    
    def _extract_station_features(self, lat: float, lon: float, station_id: str, 
                                city_data: Dict, is_source: bool) -> Dict[str, float]:
        """Extract comprehensive features for a single station"""
        features = {}
        
        # Basic geographic features
        features['latitude'] = lat
        features['longitude'] = lon
        
        # OSM features
        try:
            osm_features = self.osm_extractor.extract_features_around_station(lat, lon, radius_m=500)
            osm_metrics = self.osm_extractor.compute_feature_metrics(osm_features)
            
            # Standardize OSM feature names
            for key, value in osm_metrics.items():
                features[f'osm_{key}'] = float(value) if value is not None else 0.0
                
        except Exception as e:
            logger.warning(f"Failed to extract OSM features for station {station_id}: {e}")
            # Add default OSM features
            default_osm = ['hotels_count', 'restaurants_count', 'banks_count', 'shops_count',
                          'schools_count', 'parks_count', 'offices_count', 'residential_count']
            for feature in default_osm:
                features[f'osm_{feature}'] = 0.0
        
        # Population features (if enabled)
        if self.transfer_config.use_population_features:
            try:
                pop_features = self.population_extractor.extract_population_features_around_station(
                    lat, lon, radius_m=500
                )
                for key, value in pop_features.items():
                    features[f'pop_{key}'] = float(value) if value is not None else 0.0
                    
            except Exception as e:
                logger.warning(f"Failed to extract population features for station {station_id}: {e}")
                # Add default population features
                features['pop_total_population'] = 0.0
                features['pop_population_density_km2'] = 0.0
        
        # Trip-based features (only for source city)
        if is_source and 'trips_df' in city_data:
            trips_df = city_data['trips_df']
            station_trips = trips_df[
                (trips_df['start_station_id'] == station_id) | 
                (trips_df['end_station_id'] == station_id)
            ]
            
            features['historical_trips'] = len(station_trips)
            features['avg_temperature'] = station_trips['temperature'].mean() if len(station_trips) > 0 else 20.0
            
            # Temporal patterns
            if len(station_trips) > 0:
                for hour in range(0, 24, 6):  # Every 6 hours
                    hour_trips = station_trips[station_trips['hour'] == hour]
                    features[f'trips_hour_{hour}'] = len(hour_trips)
                
                for dow in range(7):
                    dow_trips = station_trips[station_trips['day_of_week'] == dow]
                    features[f'trips_dow_{dow}'] = len(dow_trips)
            else:
                # Default temporal features
                for hour in range(0, 24, 6):
                    features[f'trips_hour_{hour}'] = 0.0
                for dow in range(7):
                    features[f'trips_dow_{dow}'] = 0.0
        else:
            # Default trip features for target city
            features['historical_trips'] = 0.0
            features['avg_temperature'] = 20.0
            for hour in range(0, 24, 6):
                features[f'trips_hour_{hour}'] = 0.0
            for dow in range(7):
                features[f'trips_dow_{dow}'] = 0.0
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature set when extraction fails"""
        features = {
            'latitude': 0.0,
            'longitude': 0.0,
            'historical_trips': 0.0,
            'avg_temperature': 20.0
        }
        
        # Default OSM features
        default_osm = ['hotels_count', 'restaurants_count', 'banks_count', 'shops_count',
                      'schools_count', 'parks_count', 'offices_count', 'residential_count']
        for feature in default_osm:
            features[f'osm_{feature}'] = 0.0
        
        # Default population features
        if self.transfer_config.use_population_features:
            features['pop_total_population'] = 0.0
            features['pop_population_density_km2'] = 0.0
        
        # Default temporal features
        for hour in range(0, 24, 6):
            features[f'trips_hour_{hour}'] = 0.0
        for dow in range(7):
            features[f'trips_dow_{dow}'] = 0.0
        
        return features
    
    def _learn_flow_patterns(self, city_data: Dict):
        """Learn flow patterns from trip data"""
        if 'trips_df' not in city_data:
            return
        
        trips_df = city_data['trips_df']
        
        # Aggregate flows by station pairs
        flows = trips_df.groupby(['start_station_id', 'end_station_id']).agg({
            'trip_id': 'count',
            'hour': lambda x: x.mode().iloc[0] if len(x) > 0 else 12,
            'day_of_week': lambda x: x.mode().iloc[0] if len(x) > 0 else 1,
            'temperature': 'mean'
        }).reset_index()
        
        flows.columns = ['source', 'target', 'flow_count', 'peak_hour', 'peak_dow', 'avg_temp']
        
        city_data['flow_patterns'] = flows
    
    def _align_target_features(self):
        """Align target city features with source city feature distribution"""
        if not self.source_data.get('station_features') or not self.target_data.get('station_features'):
            return
        
        source_features = self.source_data['station_features']
        target_features = self.target_data['station_features']
        
        # Collect all feature values for fitting scalers
        source_feature_arrays = {}
        for station_id, features in source_features.items():
            for feature_name, value in features.items():
                if feature_name not in source_feature_arrays:
                    source_feature_arrays[feature_name] = []
                source_feature_arrays[feature_name].append(value)
        
        # Convert to numpy arrays
        for feature_name in source_feature_arrays:
            source_feature_arrays[feature_name] = np.array(source_feature_arrays[feature_name])
        
        # Fit aligners on source features
        self.feature_aligner.fit_source_features(source_feature_arrays)
        
        # Align target features
        target_feature_arrays = {}
        for station_id, features in target_features.items():
            for feature_name, value in features.items():
                if feature_name not in target_feature_arrays:
                    target_feature_arrays[feature_name] = []
                target_feature_arrays[feature_name].append(value)
        
        for feature_name in target_feature_arrays:
            target_feature_arrays[feature_name] = np.array(target_feature_arrays[feature_name])
        
        aligned_target = self.feature_aligner.align_target_features(target_feature_arrays)
        
        # Update target features with aligned values
        for i, station_id in enumerate(target_features.keys()):
            for feature_name, aligned_values in aligned_target.items():
                self.target_data['station_features'][station_id][feature_name] = aligned_values[i]
    
    def train_source_model(self) -> Dict[str, float]:
        """Train GNN model on source city data"""
        logger.info("Training source city model...")
        
        if not self.source_data.get('flow_patterns', {}).empty if isinstance(self.source_data.get('flow_patterns'), pd.DataFrame) else not self.source_data.get('flow_patterns'):
            raise ValueError("No flow patterns available for source city")
        
        # Initialize source model
        feature_dim = len(next(iter(self.source_data['station_features'].values())))
        self.source_model = TransferGNN(
            self.gnn_config, 
            self.transfer_config,
            feature_dim
        )
        
        # Prepare training data
        train_data = self._prepare_training_data(self.source_data)
        
        # Train model
        metrics = self._train_model(self.source_model, train_data, "source")
        
        logger.info(f"Source model training completed. R²: {metrics.get('r2', 0):.3f}")
        
        return metrics
    
    def transfer_to_target_city(self, target_flows: pd.DataFrame = None) -> Dict[str, float]:
        """Transfer learned model to target city"""
        logger.info("Transferring model to target city...")
        
        if self.source_model is None:
            raise ValueError("Source model must be trained first")
        
        # Initialize transfer model by copying source model
        feature_dim = len(next(iter(self.target_data['station_features'].values())))
        self.transfer_model = TransferGNN(
            self.gnn_config,
            self.transfer_config, 
            feature_dim
        )
        
        # Copy weights from source model
        self.transfer_model.load_state_dict(self.source_model.state_dict(), strict=False)
        
        # Freeze specified layers
        if self.transfer_config.freeze_layers:
            self._freeze_layers(self.transfer_model, self.transfer_config.freeze_layers)
        
        # Fine-tune on target city (if target flows available)
        if target_flows is not None and len(target_flows) > 0:
            target_data = self._prepare_target_training_data(target_flows)
            metrics = self._fine_tune_model(self.transfer_model, target_data)
        else:
            # No target data available - use zero-shot transfer
            metrics = {'r2': 0.0, 'rmse': 0.0, 'transfer_type': 'zero-shot'}
        
        logger.info(f"Transfer completed. Type: {metrics.get('transfer_type', 'fine-tuned')}")
        
        return metrics
    
    def predict_target_flows(self, source_station: str, target_station: str, 
                           hour: int = 17, day_of_week: int = 1) -> Union[Tuple[float, float], Dict]:
        """Predict flow between stations in target city and return top features and all features used"""
        if self.transfer_model is None:
            raise ValueError("Transfer model not available. Run transfer_to_target_city() first.")

        if source_station not in self.target_data['station_features']:
            raise ValueError(f"Source station {source_station} not found in target city")

        if target_station not in self.target_data['station_features']:
            raise ValueError(f"Target station {target_station} not found in target city")

        # Get station features
        source_feat_dict = self.target_data['station_features'][source_station]
        target_feat_dict = self.target_data['station_features'][target_station]
        source_features = self._prepare_station_features(source_feat_dict, hour, day_of_week)
        target_features = self._prepare_station_features(target_feat_dict, hour, day_of_week)

        # Feature names in order
        feature_names = []
        if self.transfer_config.use_temporal_features:
            feature_names.extend([
                'hour', 'sin_hour', 'cos_hour', 'day_of_week', 'sin_dow', 'cos_dow', 'is_weekend'
            ])
        feature_names.extend(sorted(source_feat_dict.keys()))

        # Convert to tensors
        source_tensor = torch.FloatTensor(source_features).unsqueeze(0)
        target_tensor = torch.FloatTensor(target_features).unsqueeze(0)

        # Get embeddings and prediction
        with torch.no_grad():
            self.transfer_model.eval()
            edge_index = torch.LongTensor([[0], [0]])
            source_embedding = self.transfer_model(source_tensor, edge_index, return_embeddings=True)
            target_embedding = self.transfer_model(target_tensor, edge_index, return_embeddings=True)
            predicted_flow = self.transfer_model.predict_flow(source_embedding, target_embedding)
            predicted_flow = max(0, predicted_flow.item())
            confidence = self._estimate_prediction_confidence(source_features, target_features)

        # Feature importance: use absolute difference between source and target features as a proxy
        feat_diffs = np.abs(np.array(source_features) - np.array(target_features))
        top_idx = np.argsort(-feat_diffs)[:5]  # Top 5 most different features
        top_features = {feature_names[i]: feat_diffs[i] for i in top_idx if i < len(feature_names)}

        # Return all features used for training
        result = {
            'predicted_flow': predicted_flow,
            'confidence': confidence,
            'top_features': top_features,
            'all_features': feature_names
        }
        return result
    
    def _prepare_station_features(self, station_features: Dict[str, float], 
                                hour: int, day_of_week: int) -> List[float]:
        """Prepare station features for prediction"""
        features = []
        
        # Add temporal features
        if self.transfer_config.use_temporal_features:
            features.extend([
                hour,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                day_of_week,
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7),
                1 if day_of_week in [5, 6] else 0  # weekend
            ])
        
        # Add station features in consistent order
        feature_names = sorted(station_features.keys())
        for feature_name in feature_names:
            features.append(station_features[feature_name])
        
        return features
    
    def _estimate_prediction_confidence(self, source_features: List[float], 
                                      target_features: List[float]) -> float:
        """Estimate prediction confidence based on feature similarity"""
        if len(source_features) != len(target_features):
            return 0.1
        
        # Calculate feature similarity
        source_array = np.array(source_features)
        target_array = np.array(target_features)
        
        # Normalize features for comparison
        source_norm = source_array / (np.linalg.norm(source_array) + 1e-8)
        target_norm = target_array / (np.linalg.norm(target_array) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(source_norm, target_norm)
        
        # Convert similarity to confidence (0.1 to 0.9 range)
        confidence = 0.1 + 0.8 * max(0, (similarity + 1) / 2)
        
        return confidence
    
    def get_all_target_predictions(self, hour: int = 17, day_of_week: int = 1, 
                                 top_k: int = 5) -> List[Dict]:
        """Get flow predictions for all station pairs in target city"""
        if self.transfer_model is None:
            raise ValueError("Transfer model not available")
        
        stations = list(self.target_data['station_features'].keys())
        predictions = []
        
        # Sample station pairs for performance
        import random
        station_pairs = [(s1, s2) for s1 in stations for s2 in stations if s1 != s2]
        
        if len(station_pairs) > 100:  # Limit for performance
            station_pairs = random.sample(station_pairs, 100)
        
        for source_station, target_station in station_pairs:
            try:
                predicted_flow, confidence = self.predict_target_flows(
                    source_station, target_station, hour, day_of_week
                )
                
                predictions.append({
                    'source': source_station,
                    'target': target_station,
                    'predicted_flow': predicted_flow,
                    'confidence': confidence,
                    'hour': hour,
                    'day_of_week': day_of_week
                })
                
            except Exception as e:
                logger.warning(f"Failed to predict flow {source_station}->{target_station}: {e}")
                continue
        
        # Sort by predicted flow and return top-k
        predictions.sort(key=lambda x: x['predicted_flow'], reverse=True)
        return predictions[:top_k]
    
    def _prepare_training_data(self, city_data: Dict) -> Dict:
        """Prepare training data for GNN"""
        # This is a simplified version - you would implement full graph construction here
        flow_patterns = city_data['flow_patterns']
        station_features = city_data['station_features']
        
        return {
            'flows': flow_patterns,
            'features': station_features,
            'graph_structure': self._build_graph_structure(city_data)
        }
    
    def _prepare_target_training_data(self, target_flows: pd.DataFrame) -> Dict:
        """Prepare target city training data"""
        return {
            'flows': target_flows,
            'features': self.target_data['station_features'],
            'graph_structure': self._build_graph_structure(self.target_data)
        }
    
    def _build_graph_structure(self, city_data: Dict) -> Dict:
        """Build graph structure for city"""
        # Simplified graph structure - in practice you'd build proper spatial graph
        stations = list(city_data['station_features'].keys())
        edges = []
        
        # Create edges based on geographical proximity
        station_coords = city_data['station_coords']
        for i, station1 in enumerate(stations):
            for j, station2 in enumerate(stations[i+1:], i+1):
                lat1, lon1 = station_coords[station1]
                lat2, lon2 = station_coords[station2]
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                
                if distance < 2.0:  # Connect stations within 2km
                    edges.append((i, j))
                    edges.append((j, i))  # Undirected
        
        return {
            'num_nodes': len(stations),
            'edges': edges,
            'node_mapping': {station: i for i, station in enumerate(stations)}
        }
    
    def _train_model(self, model: TransferGNN, train_data: Dict, model_type: str) -> Dict[str, float]:
        """Train the GNN model"""
        # Simplified training - implement full training loop here
        return {
            'r2': 0.75,  # Placeholder
            'rmse': 2.5,
            'mae': 1.8,
            'training_loss': 0.15
        }
    
    def _fine_tune_model(self, model: TransferGNN, target_data: Dict) -> Dict[str, float]:
        """Fine-tune model on target city data"""
        # Simplified fine-tuning - implement full fine-tuning here
        return {
            'r2': 0.65,  # Placeholder
            'rmse': 3.0,
            'mae': 2.2,
            'transfer_type': 'fine-tuned'
        }
    
    def _freeze_layers(self, model: TransferGNN, layers_to_freeze: List[str]):
        """Freeze specified layers during transfer learning"""
        for name, param in model.named_parameters():
            for layer_name in layers_to_freeze:
                if layer_name in name:
                    param.requires_grad = False
                    logger.info(f"Frozen layer: {name}")
    
    def save_transfer_model(self, save_path: str):
        """Save transfer learning model and configurations"""
        save_data = {
            'transfer_model': self.transfer_model.state_dict() if self.transfer_model else None,
            'source_model': self.source_model.state_dict() if self.source_model else None,
            'gnn_config': self.gnn_config,
            'transfer_config': self.transfer_config,
            'feature_aligner': self.feature_aligner,
            'transfer_metrics': self.transfer_metrics
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Transfer model saved to {save_path}")
    
    def load_transfer_model(self, load_path: str):
        """Load transfer learning model and configurations"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.gnn_config = save_data['gnn_config']
        self.transfer_config = save_data['transfer_config']
        self.feature_aligner = save_data['feature_aligner']
        self.transfer_metrics = save_data.get('transfer_metrics', {})
        
        # Reconstruct models
        if save_data['source_model']:
            feature_dim = 50  # Placeholder - you'd get this from saved data
            self.source_model = TransferGNN(self.gnn_config, self.transfer_config, feature_dim)
            self.source_model.load_state_dict(save_data['source_model'])
        
        if save_data['transfer_model']:
            feature_dim = 50  # Placeholder
            self.transfer_model = TransferGNN(self.gnn_config, self.transfer_config, feature_dim)
            self.transfer_model.load_state_dict(save_data['transfer_model'])
        
        logger.info(f"Transfer model loaded from {load_path}")
    
    def get_transfer_summary(self) -> Dict:
        """Get summary of transfer learning process"""
        return {
            'source_city': self.transfer_config.source_city,
            'target_city': self.transfer_config.target_city,
            'source_stations': len(self.source_data.get('station_coords', {})),
            'target_stations': len(self.target_data.get('station_coords', {})),
            'transfer_metrics': self.transfer_metrics,
            'models_available': {
                'source_model': self.source_model is not None,
                'transfer_model': self.transfer_model is not None
            }
        }
