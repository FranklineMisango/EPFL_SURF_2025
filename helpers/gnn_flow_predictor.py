import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv, GINConv, TransformerConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
import os
import time
from typing import Dict, List, Tuple, Optional
import networkx as nx
from dataclasses import dataclass
import pickle

logger = logging.getLogger(__name__)

# Limit CPU multithreading to reduce the chance of native-level crashes
# (OpenMP / MKL issues can cause segmentation faults in some environments).
try:
    # Prefer setting via torch if available
    import torch as _torch_temp
    _torch_temp.set_num_threads(1)
    _torch_temp.set_num_interop_threads(1)
except Exception:
    pass

@dataclass
class GNNConfig:
    """Configuration for GNN model"""
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    attention_heads: int = 4
    gnn_type: str = "GCN"  # Options: "GCN", "GAT", "GraphSAGE", "GIN", "Transformer", "GGNN", "DCRNN"
    edge_features: bool = True
    use_batch_norm: bool = True
    use_residual: bool = False
    # Time conditioning / training controls
    time_emb_dim: int = 2  # simple numeric time vector (hour_norm, dow_norm) by default
    log_labels: bool = True
    max_train_samples: int = 20000
    negative_sample_ratio: float = 0.5

class BikeFlowGNN(nn.Module):
    """Graph Neural Network for bike flow prediction between stations"""
    
    def __init__(self, config: GNNConfig, node_features: int, edge_features: int = 0):
        super(BikeFlowGNN, self).__init__()
        self.config = config
        self.node_features = node_features
        self.edge_features = edge_features
        self.gnn_type = config.gnn_type
        
        # Node embedding layers
        self.node_embedding = nn.Linear(node_features, config.hidden_dim)
        
        # Edge embedding (if using edge features)
        if edge_features > 0 and config.edge_features:
            self.edge_embedding = nn.Linear(edge_features, config.hidden_dim // 4)
        
        # Graph convolution layers based on type
        self.conv_layers = nn.ModuleList()
        # conv_output_dims will store the output feature size of each conv layer
        # (needed so BatchNorm1d can be created with the correct number of features)
        self.conv_output_dims: List[int] = self._build_conv_layers()

        # Batch normalization layers (if enabled).
        # Create them after conv layers so sizes match (important for GAT concat heads)
        self.batch_norms = nn.ModuleList()
        if config.use_batch_norm:
            for dim in self.conv_output_dims:
                self.batch_norms.append(nn.BatchNorm1d(dim))
        
        # Output layers for flow prediction
        # Determine final node embedding dimension produced by the conv layers. Use
        # the last entry in conv_output_dims if conv layers exist, otherwise fall
        # back to config.hidden_dim. This ensures the predictor input size matches
        # the runtime node embedding size (important for GAT with concat heads).
        final_node_dim = self.conv_output_dims[-1] if len(self.conv_output_dims) > 0 else config.hidden_dim
        # *2 for source + target concatenation, plus time embedding dim if present
        predictor_input_dim = final_node_dim * 2 + getattr(config, 'time_emb_dim', 0)
        self.flow_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)  # Single output for flow prediction
        )

        self.dropout = nn.Dropout(config.dropout)
        
    def _build_conv_layers(self):
        """Build convolution layers based on GNN type"""
        config = self.config
        output_dims: List[int] = []

        if self.gnn_type == "GCN":
            # Graph Convolutional Network - simple and robust
            for i in range(config.num_layers):
                in_channels = config.hidden_dim
                out_channels = config.hidden_dim
                self.conv_layers.append(GCNConv(in_channels, out_channels))
                output_dims.append(out_channels)

        elif self.gnn_type == "GAT":
            # Graph Attention Network - handle multi-head concatenation
            for i in range(config.num_layers):
                if i == 0:
                    in_channels = config.hidden_dim
                else:
                    # If previous layer used concat heads, its output will be hidden_dim * heads
                    prev_concat = True if i - 1 < config.num_layers - 1 else False
                    in_channels = config.hidden_dim * config.attention_heads if prev_concat else config.hidden_dim

                out_channels = config.hidden_dim
                # Use concat=True on intermediate layers to increase capacity, but final layer typically concat=False
                concat = True if i < config.num_layers - 1 else False
                heads = config.attention_heads if concat else 1

                self.conv_layers.append(
                    GATConv(in_channels, out_channels, heads=heads, dropout=config.dropout, concat=concat)
                )

                # The output dimension depends on concat
                out_dim = out_channels * heads if concat else out_channels
                output_dims.append(out_dim)

        elif self.gnn_type == "GraphSAGE":
            # GraphSAGE - good for larger graphs and inductive learning
            for i in range(config.num_layers):
                in_channels = config.hidden_dim
                out_channels = config.hidden_dim
                self.conv_layers.append(SAGEConv(in_channels, out_channels, aggr='mean'))
                output_dims.append(out_channels)

        elif self.gnn_type == "GGNN":
            # Gated Graph Neural Network (requires DGL or custom implementation)
            try:
                from dgl.nn.pytorch import GatedGraphConv
            except ImportError:
                raise ImportError("DGL is required for GGNN. Please install dgl.")
            for i in range(config.num_layers):
                in_channels = config.hidden_dim
                out_channels = config.hidden_dim
                # GGNN uses a single GatedGraphConv layer with n_steps
                self.conv_layers.append(GatedGraphConv(out_channels, n_steps=3))
                output_dims.append(out_channels)

        elif self.gnn_type == "DCRNN":
            # DCRNN is a spatio-temporal model and requires a custom implementation or pytorch-dcrnn
            # This is a stub for future integration.
            raise NotImplementedError("DCRNN integration requires sequence data and a custom model. See pytorch-dcrnn.")

        elif self.gnn_type == "GIN":
            # Graph Isomorphism Network
            for i in range(config.num_layers):
                in_channels = config.hidden_dim
                out_channels = config.hidden_dim
                nn_func = nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
                self.conv_layers.append(GINConv(nn_func))
                output_dims.append(out_channels)

        elif self.gnn_type == "Transformer":
            # Graph Transformer
            for i in range(config.num_layers):
                in_channels = config.hidden_dim
                out_channels = config.hidden_dim
                heads = config.attention_heads
                self.conv_layers.append(TransformerConv(in_channels, out_channels, heads=heads, dropout=config.dropout))
                output_dims.append(out_channels * heads)

        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")

        return output_dims
    
    def forward(self, x, edge_index, edge_attr=None, source_nodes=None, target_nodes=None, time_feats=None):
        """
        Forward pass with support for different GNN architectures
        """
        # Node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Store input for residual connections
        residual_input = x if self.config.use_residual else None
        
        # Graph convolution layers
        for i, conv in enumerate(self.conv_layers):
            x_prev = x
            
            # Apply convolution based on type
            if self.gnn_type == "GCN":
                x = conv(x, edge_index)
            elif self.gnn_type == "GAT":
                x = conv(x, edge_index)
            elif self.gnn_type == "GraphSAGE":
                x = conv(x, edge_index)
            
            # Apply batch normalization if enabled
            if self.config.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Apply activation and dropout (except on last layer)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                
                # Residual connection
                if self.config.use_residual and x.shape == x_prev.shape:
                    x = x + x_prev
                    
                x = self.dropout(x)
        
        # If predicting flows between specific node pairs
        if source_nodes is not None and target_nodes is not None:
            source_embeddings = x[source_nodes]  # [batch_size, hidden_dim]
            target_embeddings = x[target_nodes]  # [batch_size, hidden_dim]
            
            # Concatenate source and target embeddings
            if time_feats is not None:
                # time_feats expected shape: [batch_size, time_emb_dim]
                flow_input = torch.cat([source_embeddings, target_embeddings, time_feats], dim=1)
            else:
                flow_input = torch.cat([source_embeddings, target_embeddings], dim=1)
            
            # Predict flow
            # Sanity check: ensure flow_input width matches predictor's expected in_features
            try:
                expected_in = self.flow_predictor[0].in_features
            except Exception:
                expected_in = None

            if expected_in is not None and flow_input.size(1) != expected_in:
                logger.error(f"Flow predictor input mismatch: flow_input.shape={flow_input.shape}, expected_in={expected_in}")
                raise RuntimeError(f"Flow predictor input width ({flow_input.size(1)}) does not match expected ({expected_in}).")

            flow_prediction = self.flow_predictor(flow_input)
            return flow_prediction.squeeze(-1)  # [batch_size]
        
        return x  # Return node embeddings

class GNNFlowPredictor:
    """Enhanced GNN-based bike flow predictor with comprehensive OSM caching"""
    
    def __init__(self, trips_df, config: Optional[GNNConfig] = None, use_cached_features=True, cache_file: Optional[str] = None):
        self.trips_df = trips_df
        self.config = config or GNNConfig()
        self.use_cached_features = use_cached_features
        # Optional explicit cache file to load (overrides auto-detection)
        self.explicit_cache_file = cache_file
        
        # Initialize components
        self.station_coords = {}
        self.station_features = {}
        self.osm_features = {}
        self.population_features = {}
        self.comprehensive_features = {}
        self.cached_osm_df = None
        
        # GNN-specific components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Graph structure
        self.graph_data = None
        self.node_feature_matrix = None
        self.edge_index = None
        self.edge_features = None
        
        # Feature caching
        self.cache_dir = "cache/gnn_features"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load cached OSM features if available
        if self.use_cached_features:
            self.load_cached_osm_features()
        
        logger.info(f"GNN Flow Predictor initialized on device: {self.device}")
        if self.cached_osm_df is not None:
            logger.info(f"Loaded cached OSM features: {len(self.cached_osm_df)} stations, {len(self.cached_osm_df.columns)-3} features")
    
    def load_cached_osm_features(self):
        """Load cached OSM features from the master cache system"""
        # Prefer explicit cache_file if provided, then canonical ML-ready file, then auto-detect
        canonical = "cache/ml_ready/station_features.csv"
        # If caller supplied an explicit cache_file, prefer that
        if getattr(self, 'explicit_cache_file', None):
            candidate = self.explicit_cache_file
            if not os.path.exists(candidate):
                logger.warning(f"Explicit cache file provided but not found: {candidate}")
                candidate = None
        else:
            candidate = None
        ml_ready_dir = os.path.join("cache", "ml_ready")

        # If an explicit cache file was provided and exists, prefer it
        if getattr(self, 'explicit_cache_file', None) and os.path.exists(self.explicit_cache_file):
            candidate = self.explicit_cache_file
            logger.info(f"Using explicit cached OSM features file: {candidate}")

        # 1) canonical path (only considered if explicit cache_file was not set)
        if candidate is None and os.path.exists(canonical):
            candidate = canonical

        # 2) look for region/radius specific consolidated files (e.g. switzerland_station_features_500m.csv)
        if candidate is None and os.path.isdir(ml_ready_dir):
            try:
                # prefer files that contain 'station_features' and optionally a radius
                files = [f for f in os.listdir(ml_ready_dir) if f.endswith('.csv')]
                # rank candidates: exact 'station_features.csv' (already checked), then '*station_features*', then any csv
                station_feature_files = [f for f in files if 'station_features' in f]
                if station_feature_files:
                    # prefer files that contain a numeric radius (e.g. '500m') else pick first
                    radius_files = [f for f in station_feature_files if any(ch.isdigit() for ch in f)]
                    pick = radius_files[0] if radius_files else station_feature_files[0]
                    candidate = os.path.join(ml_ready_dir, pick)
                elif files:
                    candidate = os.path.join(ml_ready_dir, files[0])
            except Exception:
                candidate = None

        if candidate is None:
            logger.warning(f"❌ Cached OSM features not found in {ml_ready_dir}")
            logger.info("� Run 'python setup_osm_cache.py download' to create OSM feature cache")
            self.cached_osm_df = None
            return

        # Attempt to load the detected candidate
        try:
            self.cached_osm_df = pd.read_csv(candidate)
            logger.info(f"✅ Loaded cached OSM features from {candidate}")
            # columns may include station_id, lat, lon and many features
            approx_feature_count = max(0, len(self.cached_osm_df.columns) - 3)
            logger.info(f"📊 Features: {len(self.cached_osm_df)} stations, {approx_feature_count} features")

            # Log a short summary of feature categories found
            feature_categories = {
                'Traditional OSM': len([c for c in self.cached_osm_df.columns if c.startswith('trad_')]),
                'Population': len([c for c in self.cached_osm_df.columns if c.startswith('pop_')]),
                'Comprehensive OSM': len([c for c in self.cached_osm_df.columns if c.startswith('comp_')])
            }
            for category, count in feature_categories.items():
                if count > 0:
                    logger.info(f"   {category}: {count} features")

        except Exception as e:
            logger.warning(f"❌ Failed to load cached OSM features ({candidate}): {e}")
            logger.info("💡 Run 'python setup_osm_cache.py download' to create OSM feature cache")
            self.cached_osm_df = None
    
    def build_station_network(self):
        """Build station coordinate mapping from trips data or cached features"""
        
        if self.cached_osm_df is not None:
            # Use coordinates from cached OSM features
            logger.info("🗺️ Building station network from cached OSM features...")
            for _, row in self.cached_osm_df.iterrows():
                station_id = row['station_id']
                self.station_coords[station_id] = (row['lat'], row['lon'])
        else:
            # Fallback to trips data - parse coordinate strings
            logger.info("🗺️ Building station network from trips data...")
            
            # Parse start coordinates
            start_stations = self.trips_df.groupby('start_station_id')['start_coords'].first().reset_index()
            for _, row in start_stations.iterrows():
                station_id = row['start_station_id']
                coords_str = row['start_coords']
                try:
                    # Parse coordinate string (format: "(lat, lon)")
                    coords_str = coords_str.strip('()')
                    lat, lon = map(float, coords_str.split(', '))
                    self.station_coords[station_id] = (lat, lon)
                except Exception as e:
                    logger.warning(f"Failed to parse start coordinates for station {station_id}: {coords_str}")
            
            # Parse end coordinates
            end_stations = self.trips_df.groupby('end_station_id')['end_coords'].first().reset_index()
            for _, row in end_stations.iterrows():
                station_id = row['end_station_id']
                if station_id not in self.station_coords:  # Only add if not already present
                    coords_str = row['end_coords']
                    try:
                        # Parse coordinate string (format: "(lat, lon)")
                        coords_str = coords_str.strip('()')
                        lat, lon = map(float, coords_str.split(', '))
                        self.station_coords[station_id] = (lat, lon)
                    except Exception as e:
                        logger.warning(f"Failed to parse end coordinates for station {station_id}: {coords_str}")
        
        logger.info(f"Built network with {len(self.station_coords)} stations")
    
    def download_and_cache_all_osm_features(self, force_refresh=False):
        """Download and cache comprehensive OSM features for all stations"""
        
        # Check if cached features are already available
        if self.cached_osm_df is not None and not force_refresh:
            logger.info("✅ Using cached OSM features from master cache system")
            logger.info(f"📊 Available: {len(self.cached_osm_df)} stations with {len(self.cached_osm_df.columns)-3} features")
            
            # Convert cached DataFrame to the format expected by the rest of the system
            for _, row in self.cached_osm_df.iterrows():
                station_id = row['station_id']
                features = {}
                
                # Extract all non-coordinate features
                for col in self.cached_osm_df.columns:
                    if col not in ['station_id', 'lat', 'lon']:
                        features[col] = row[col]
                
                self.osm_features[station_id] = features
            
            logger.info(f"✅ Converted cached features for {len(self.osm_features)} stations")
            return
        
        # Fallback to original download logic if no cache or force refresh
        logger.warning("No cached OSM features available, falling back to download")
        logger.info("💡 Consider running 'python setup_osm_cache.py download' to create a comprehensive cache")
        
        cache_file = os.path.join(self.cache_dir, "all_osm_features.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            logger.info("Loading cached OSM features...")
            with open(cache_file, 'r') as f:
                self.osm_features = json.load(f)
                # Convert string keys back to appropriate types
                self.osm_features = {eval(k) if k.replace('.', '').isdigit() else k: v 
                                   for k, v in self.osm_features.items()}
            logger.info(f"Loaded OSM features for {len(self.osm_features)} stations")
            return
        
        logger.info("Downloading comprehensive OSM features for all stations...")
        
        try:
            # Use comprehensive OSM downloader
            from comprehensive_osm_downloader import ComprehensiveOSMDownloader, OSMDownloadConfig
            
            # Configure comprehensive download
            download_config = OSMDownloadConfig(
                cache_dir=os.path.join(self.cache_dir, "comprehensive_osm"),
                radius_meters=1000,
                use_osmnx=True,
                use_overpass=True,
                rate_limit_seconds=0.5,
                batch_size=5,
                force_refresh=force_refresh
            )
            
            downloader = ComprehensiveOSMDownloader(download_config)
            self.osm_features = downloader.download_all_features(self.station_coords)
            
        except ImportError:
            logger.warning("Comprehensive OSM downloader not available, falling back to traditional method")
            self._download_osm_features_traditional()
        except Exception as e:
            logger.error(f"Comprehensive OSM download failed: {e}")
            logger.info("Falling back to traditional OSM download...")
            self._download_osm_features_traditional()
        
        # Save final cache
        self._save_osm_cache(cache_file)
        logger.info(f"Downloaded and cached comprehensive OSM features for {len(self.osm_features)} stations")
    
    def _download_osm_features_traditional(self):
        """Download OSM features using traditional Overpass API approach"""
        from osm_feature_extractor import OSMFeatureExtractor
        
        osm_extractor = OSMFeatureExtractor()
        total_stations = len(self.station_coords)
        processed = 0
        
        for station_id, (lat, lon) in self.station_coords.items():
            try:
                # Extract features with larger radius for comprehensive coverage
                features = osm_extractor.extract_features_around_station(
                    lat, lon, radius_m=1000  # Larger radius for more comprehensive features
                )
                
                # Compute detailed metrics
                metrics = osm_extractor.compute_feature_metrics(features)
                self.osm_features[station_id] = metrics
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_stations} stations (traditional)")
                
                # Rate limiting to respect API limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Failed to extract OSM features for station {station_id}: {e}")
                if station_id not in self.osm_features:
                    self.osm_features[station_id] = {}
    
    def _download_osm_features_osmnx(self):
        """Download comprehensive OSM features using OSMnx for enhanced coverage"""
        try:
            import osmnx as ox
            import geopandas as gpd
            from shapely.geometry import Point
            
            logger.info("Downloading OSM data using OSMnx...")
            
            # Configure OSMnx
            ox.config(use_cache=True, log_console=False)
            
            # Get bounding box for all stations
            lats = [coord[0] for coord in self.station_coords.values()]
            lons = [coord[1] for coord in self.station_coords.values()]
            
            # Add buffer for comprehensive coverage
            buffer = 0.01  # ~1km buffer
            bbox = (min(lats) - buffer, max(lats) + buffer, min(lons) - buffer, max(lons) + buffer)
            
            # Download different types of OSM features
            osm_features_config = {
                'amenities': ['restaurant', 'cafe', 'bar', 'fast_food', 'pub', 'bank', 'atm', 'pharmacy', 'hospital', 'clinic', 'school', 'university', 'library', 'post_office', 'police', 'fire_station'],
                'shops': ['convenience', 'supermarket', 'clothes', 'shoes', 'electronics', 'books', 'bakery', 'butcher', 'car_repair', 'bicycle'],
                'leisure': ['park', 'playground', 'sports_centre', 'swimming_pool', 'fitness_centre', 'cinema', 'theatre', 'museum', 'gallery'],
                'tourism': ['hotel', 'hostel', 'guest_house', 'attraction', 'museum', 'gallery', 'viewpoint'],
                'public_transport': ['bus_stop', 'tram_stop', 'subway_entrance', 'railway_station'],
                'landuse': ['residential', 'commercial', 'industrial', 'retail', 'recreation_ground']
            }
            
            # Download POIs for each category
            all_pois = {}
            for category, tags in osm_features_config.items():
                try:
                    logger.info(f"Downloading {category} features...")
                    
                    # Create tags dict for OSMnx
                    osm_tags = {}
                    if category == 'landuse':
                        osm_tags = {'landuse': tags}
                    else:
                        osm_tags = {category[:-1] if category.endswith('s') else category: tags}
                    
                    # Download POIs
                    pois = ox.geometries_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], osm_tags)
                    
                    if not pois.empty:
                        all_pois[category] = pois
                        logger.info(f"Downloaded {len(pois)} {category} features")
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Failed to download {category}: {e}")
                    continue
            
            # Process features for each station
            self._process_osmnx_features(all_pois)
            
        except ImportError:
            logger.warning("OSMnx not available, skipping enhanced OSM download")
        except Exception as e:
            logger.error(f"OSMnx download failed: {e}")
    
    def _process_osmnx_features(self, all_pois):
        """Process OSMnx POI data for each station"""
        import geopandas as gpd
        from shapely.geometry import Point
        
        logger.info("Processing OSMnx features for each station...")
        
        for station_id, (lat, lon) in self.station_coords.items():
            try:
                station_point = Point(lon, lat)
                station_features = self.osm_features.get(station_id, {})
                
                # Analyze features within different radii
                radii = [250, 500, 1000]  # meters
                
                for category, pois_gdf in all_pois.items():
                    if pois_gdf.empty:
                        continue
                    
                    # Ensure CRS is in meters for distance calculation
                    if pois_gdf.crs != 'EPSG:3857':
                        pois_gdf = pois_gdf.to_crs('EPSG:3857')
                        station_point_proj = gpd.GeoSeries([station_point], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
                    else:
                        station_point_proj = station_point
                    
                    for radius in radii:
                        # Create buffer around station
                        buffer = station_point_proj.buffer(radius)
                        
                        # Find POIs within buffer
                        within_buffer = pois_gdf[pois_gdf.intersects(buffer)]
                        count = len(within_buffer)
                        
                        # Store features
                        feature_key = f'osmnx_{category}_count_{radius}m'
                        station_features[feature_key] = count
                        
                        # Calculate density (features per km²)
                        area_km2 = (radius / 1000) ** 2 * 3.14159
                        density_key = f'osmnx_{category}_density_{radius}m'
                        station_features[density_key] = count / area_km2 if area_km2 > 0 else 0
                
                # Calculate diversity metrics
                total_features = sum(v for k, v in station_features.items() if k.startswith('osmnx_') and k.endswith('_count_500m'))
                unique_categories = sum(1 for k, v in station_features.items() if k.startswith('osmnx_') and k.endswith('_count_500m') and v > 0)
                
                station_features['osmnx_total_features_500m'] = total_features
                station_features['osmnx_category_diversity_500m'] = unique_categories
                station_features['osmnx_diversity_ratio_500m'] = unique_categories / len(all_pois) if len(all_pois) > 0 else 0
                
                self.osm_features[station_id] = station_features
                
            except Exception as e:
                logger.warning(f"Failed to process OSMnx features for station {station_id}: {e}")
                continue
    
    def _save_osm_cache(self, cache_file):
        """Save OSM features to cache"""
        # Convert keys to strings for JSON serialization
        cache_data = {str(k): v for k, v in self.osm_features.items()}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def download_and_cache_population_features(self, force_refresh=False):
        """Download and cache population features for all stations"""
        from population_feature_extractor import PopulationFeatureExtractor, EnhancedOSMFeatureExtractor
        
        cache_file = os.path.join(self.cache_dir, "all_population_features.json")
        
        if os.path.exists(cache_file) and not force_refresh:
            logger.info("Loading cached population features...")
            with open(cache_file, 'r') as f:
                self.population_features = json.load(f)
                self.population_features = {eval(k) if k.replace('.', '').isdigit() else k: v 
                                          for k, v in self.population_features.items()}
            logger.info(f"Loaded population features for {len(self.population_features)} stations")
            return
        
        logger.info("Downloading population features for all stations...")
        pop_extractor = PopulationFeatureExtractor()
        enhanced_extractor = EnhancedOSMFeatureExtractor()
        
        total_stations = len(self.station_coords)
        processed = 0
        
        for station_id, (lat, lon) in self.station_coords.items():
            try:
                # Extract population features
                pop_features = pop_extractor.extract_population_features_around_station(
                    lat, lon, radius_m=1000
                )
                
                # Extract comprehensive features (OSM + Population + Interactions)
                comprehensive = enhanced_extractor.extract_comprehensive_features(
                    lat, lon, radius_m=1000
                )
                
                self.population_features[station_id] = {
                    **pop_features,
                    **comprehensive
                }
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_stations} stations for population data")
                    # Save intermediate results
                    self._save_population_cache(cache_file)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to extract population features for station {station_id}: {e}")
                self.population_features[station_id] = {}
        
        # Save final cache
        self._save_population_cache(cache_file)
        logger.info(f"Downloaded and cached population features for {len(self.population_features)} stations")
    
    def _save_population_cache(self, cache_file):
        """Save population features to cache"""
        cache_data = {str(k): v for k, v in self.population_features.items()}
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def engineer_comprehensive_features(self):
        """Engineer comprehensive features for all stations using cached OSM data"""
        logger.info("Engineering comprehensive features for all stations...")
        
        for station_id in self.station_coords.keys():
            features = {}
            
            # Basic trip features
            station_trips = self.trips_df[
                (self.trips_df['start_station_id'] == station_id) | 
                (self.trips_df['end_station_id'] == station_id)
            ]
            
            # Basic features
            features['total_trips'] = len(station_trips)
            features['avg_temperature'] = station_trips['temperature'].mean() if len(station_trips) > 0 else 20.0
            features['lat'] = self.station_coords[station_id][0]
            features['lon'] = self.station_coords[station_id][1]
            
            # Temporal features - more comprehensive
            for hour in range(24):
                outflow = len(station_trips[
                    (station_trips['start_station_id'] == station_id) & 
                    (station_trips['hour'] == hour)
                ])
                inflow = len(station_trips[
                    (station_trips['end_station_id'] == station_id) & 
                    (station_trips['hour'] == hour)
                ])
                features[f'outflow_hour_{hour}'] = outflow
                features[f'inflow_hour_{hour}'] = inflow
                features[f'net_flow_hour_{hour}'] = outflow - inflow
            
            for dow in range(7):
                dow_outflow = len(station_trips[
                    (station_trips['start_station_id'] == station_id) & 
                    (station_trips['day_of_week'] == dow)
                ])
                dow_inflow = len(station_trips[
                    (station_trips['end_station_id'] == station_id) & 
                    (station_trips['day_of_week'] == dow)
                ])
                features[f'outflow_dow_{dow}'] = dow_outflow
                features[f'inflow_dow_{dow}'] = dow_inflow
                features[f'net_flow_dow_{dow}'] = dow_outflow - dow_inflow
            
            # Use cached OSM features if available
            if self.cached_osm_df is not None:
                station_osm = self.cached_osm_df[self.cached_osm_df['station_id'] == station_id]
                if not station_osm.empty:
                    station_row = station_osm.iloc[0]
                    
                    # Add all cached OSM features
                    for col in self.cached_osm_df.columns:
                        if col not in ['station_id', 'lat', 'lon']:
                            features[f'cached_{col}'] = station_row[col]
                    
                    logger.debug(f"Added {len([c for c in self.cached_osm_df.columns if c not in ['station_id', 'lat', 'lon']])} cached OSM features for station {station_id}")
            else:
                # Fallback: use legacy feature extraction if no cache available
                logger.warning(f"No cached OSM features available for station {station_id}, using legacy extraction")
                
                # OSM features (if available)
                osm_data = self.osm_features.get(station_id, {})
                for osm_feature, value in osm_data.items():
                    features[f'osm_{osm_feature}'] = value
                
                # Population features (if available)
                pop_data = self.population_features.get(station_id, {})
                for pop_feature, value in pop_data.items():
                    features[f'pop_{pop_feature}'] = value
            
            self.station_features[station_id] = features
        
        logger.info(f"Engineered features for {len(self.station_features)} stations")
        
        # Log feature summary
        if self.station_features:
            sample_features = next(iter(self.station_features.values()))
            cached_features = len([k for k in sample_features.keys() if k.startswith('cached_')])
            temporal_features = len([k for k in sample_features.keys() if 'hour_' in k or 'dow_' in k])
            basic_features = len([k for k in sample_features.keys() if not k.startswith('cached_') and 'hour_' not in k and 'dow_' not in k])
            
            logger.info(f"📊 Feature breakdown per station:")
            logger.info(f"   🏢 Basic features: {basic_features}")
            logger.info(f"   ⏰ Temporal features: {temporal_features}")
            logger.info(f"   🗺️ Cached OSM features: {cached_features}")
            logger.info(f"   📈 Total features: {len(sample_features)}")
    
    def build_graph_structure(self):
        """Build graph structure with stations as nodes and spatial/flow relationships as edges"""
        logger.info("Building graph structure...")
        
        # Create node mapping
        station_ids = list(self.station_coords.keys())
        station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
        
        # Build node feature matrix
        feature_names = set()
        for features in self.station_features.values():
            feature_names.update(features.keys())
        feature_names = sorted(list(feature_names))
        
        node_features = []
        for station_id in station_ids:
            station_feat = self.station_features[station_id]
            node_feat = [station_feat.get(feat_name, 0) for feat_name in feature_names]
            node_features.append(node_feat)
        
        self.node_feature_matrix = np.array(node_features, dtype=np.float32)
        
        # Normalize features
        self.node_feature_matrix = self.scaler.fit_transform(self.node_feature_matrix)
        
        # Build edge index based on spatial proximity and flow relationships
        edges = []
        edge_features = []
        
        # Spatial edges (connect nearby stations)
        for i, station_i in enumerate(station_ids):
            lat_i, lon_i = self.station_coords[station_i]
            
            for j, station_j in enumerate(station_ids):
                if i != j:
                    lat_j, lon_j = self.station_coords[station_j]
                    
                    # Calculate distance
                    distance = np.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2)
                    
                    # Connect if within reasonable distance (adjust threshold as needed)
                    if distance < 0.02:  # Approximately 2km in lat/lon degrees
                        edges.append([i, j])
                        
                        # Edge features: distance, elevation difference (if available)
                        edge_feat = [distance, 0]  # Can add more edge features
                        edge_features.append(edge_feat)
        
        # Flow-based edges (connect stations with historical flows)
        flow_pairs = self.trips_df.groupby(['start_station_id', 'end_station_id']).size()
        
        for (start_station, end_station), flow_count in flow_pairs.items():
            if start_station in station_to_idx and end_station in station_to_idx:
                if flow_count > 5:  # Minimum flow threshold
                    i = station_to_idx[start_station]
                    j = station_to_idx[end_station]
                    
                    # Check if edge already exists (spatial)
                    edge_exists = any(edge[0] == i and edge[1] == j for edge in edges)
                    
                    if not edge_exists:
                        edges.append([i, j])
                        
                        # Edge features: flow count, normalized flow
                        lat_i, lon_i = self.station_coords[start_station]
                        lat_j, lon_j = self.station_coords[end_station]
                        distance = np.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2)
                        
                        edge_feat = [distance, flow_count / 100.0]  # Normalize flow
                        edge_features.append(edge_feat)
        
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.edge_features = torch.tensor(edge_features, dtype=torch.float32)
        
        logger.info(f"Built graph with {len(station_ids)} nodes and {len(edges)} edges")
        
        # Create PyTorch Geometric data object
        self.graph_data = Data(
            x=torch.tensor(self.node_feature_matrix, dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=self.edge_features
        )
        
        return station_to_idx, feature_names
    
    def prepare_training_data(self):
        """Prepare training data for GNN"""
        logger.info("Preparing training data...")
        
        # Get station mapping
        station_ids = list(self.station_coords.keys())
        station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
        
        # Parse timestamps to extract hour and day of week
        trips_df_processed = self.trips_df.copy()
        
        # Convert timestamp format (20240429_160100) to datetime components
        def parse_timestamp(timestamp_str):
            try:
                import datetime
                dt = datetime.datetime.strptime(str(timestamp_str), '%Y%m%d_%H%M%S')
                return dt.hour, dt.weekday()
            except:
                return 12, 0  # Default fallback
        
        # Apply parsing
        trips_df_processed[['hour', 'day_of_week']] = trips_df_processed['start_time'].apply(
            lambda x: pd.Series(parse_timestamp(x))
        )
        
        # Create training samples
        training_samples = []
        
        # Sample from different time periods for variety
        sample_hours = list(range(0, 24, 3))  # Every 3 hours
        sample_days = list(range(7))  # All days of week

        for hour in sample_hours:
            for dow in sample_days:
                hour_trips = trips_df_processed[
                    (trips_df_processed['hour'] == hour) & 
                    (trips_df_processed['day_of_week'] == dow)
                ]

                if len(hour_trips) == 0:
                    continue

                flows = hour_trips.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='flow_count')

                for _, row in flows.iterrows():
                    start_station = row['start_station_id']
                    end_station = row['end_station_id']
                    flow_count = row['flow_count']

                    if start_station in station_to_idx and end_station in station_to_idx:
                        # time features normalized: hour/23, dow/6
                        hour_norm = hour / 23.0
                        dow_norm = dow / 6.0
                        training_samples.append({
                            'source_idx': station_to_idx[start_station],
                            'target_idx': station_to_idx[end_station],
                            'flow': flow_count,
                            'hour': hour,
                            'day_of_week': dow,
                            'time_vec': [hour_norm, dow_norm]
                        })
        
        # Optional negative sampling to balance dataset
        max_samples = getattr(self.config, 'max_train_samples', None)
        neg_ratio = getattr(self.config, 'negative_sample_ratio', 0.5)

        if max_samples is not None and len(training_samples) > max_samples:
            # sample positives and add negatives
            np.random.shuffle(training_samples)
            pos_samples = training_samples[:max_samples]
            # create simple negatives (pairs with zero flow)
            neg_needed = int(len(pos_samples) * neg_ratio)
            neg_samples = []
            station_ids = list(station_to_idx.keys())
            attempts = 0
            while len(neg_samples) < neg_needed and attempts < neg_needed * 10:
                s = np.random.choice(station_ids)
                t = np.random.choice(station_ids)
                attempts += 1
                if s != t:
                    neg_samples.append({
                        'source_idx': station_to_idx[s],
                        'target_idx': station_to_idx[t],
                        'flow': 0.0,
                        'hour': 12,
                        'day_of_week': 1,
                        'time_vec': [12/23.0, 1/6.0]
                    })
            training_samples = pos_samples + neg_samples

        logger.info(f"Prepared {len(training_samples)} training samples")
        return training_samples
    
    def train_gnn_model(self, training_samples):
        """Train the GNN model"""
        logger.info("Training GNN model...")

        # Initialize model
        node_features = self.node_feature_matrix.shape[1]
        edge_features = self.edge_features.shape[1] if getattr(self, 'edge_features', None) is not None and self.edge_features.size(0) > 0 else 0

        self.model = BikeFlowGNN(self.config, node_features, edge_features).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        # Prepare training data tensors
        source_indices = torch.tensor([sample['source_idx'] for sample in training_samples], dtype=torch.long)
        target_indices = torch.tensor([sample['target_idx'] for sample in training_samples], dtype=torch.long)
        if getattr(self.config, 'log_labels', False):
            flows = torch.tensor([np.log1p(sample['flow']) for sample in training_samples], dtype=torch.float32)
        else:
            flows = torch.tensor([sample['flow'] for sample in training_samples], dtype=torch.float32)

        time_feats = torch.tensor([sample.get('time_vec', [0.5, 0.5]) for sample in training_samples], dtype=torch.float32)

        # Train/val split
        train_size = int(0.8 * len(training_samples))
        indices = torch.randperm(len(training_samples))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Move to device
        graph_data = self.graph_data.to(self.device)
        source_indices = source_indices.to(self.device)
        target_indices = target_indices.to(self.device)
        flows = flows.to(self.device)
        time_feats = time_feats.to(self.device)

        # Training loop
        self.model.train()
        best_val_loss = float('inf')

        try:
            torch.manual_seed(0)
            torch.autograd.set_detect_anomaly(True)
        except Exception:
            pass

        try:
            for epoch in range(self.config.epochs):
                epoch_start = time.time()
                optimizer.zero_grad()

                predictions = self.model(
                    graph_data.x,
                    graph_data.edge_index,
                    getattr(graph_data, 'edge_attr', None),
                    source_indices[train_indices],
                    target_indices[train_indices],
                    time_feats[train_indices]
                )

                loss = criterion(predictions, flows[train_indices])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                if epoch % 10 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_predictions = self.model(
                            graph_data.x,
                            graph_data.edge_index,
                            getattr(graph_data, 'edge_attr', None),
                            source_indices[val_indices],
                            target_indices[val_indices],
                            time_feats[val_indices]
                        )
                        val_loss = criterion(val_predictions, flows[val_indices])

                    logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        try:
                            torch.save(self.model.state_dict(), os.path.join(self.cache_dir, 'best_gnn_model.pth'))
                        except Exception:
                            logger.warning("Failed to save best model state")

                    self.model.train()

                # Log epoch wall-clock time
                try:
                    epoch_time = time.time() - epoch_start
                    logger.info(f"Epoch {epoch} time: {epoch_time:.2f}s")
                except Exception:
                    pass

            logger.info("GNN training completed")
        except Exception as e:
            logger.error(f"Exception during GNN training: {e}")
            import traceback
            traceback.print_exc()
            try:
                torch.save(self.model.state_dict(), os.path.join(self.cache_dir, 'failed_gnn_model.pth'))
                logger.info("Saved model state after failure")
            except Exception:
                logger.warning("Failed to save model state after training exception")
    
    def predict_flows_gnn(self, source_station, target_stations, hour, day_of_week):
        """Predict flows using trained GNN"""
        if self.model is None:
            logger.warning("Model not trained")
            return {}
        
        station_ids = list(self.station_coords.keys())
        station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
        
        if source_station not in station_to_idx:
            return {}
        
        self.model.eval()
        predictions = {}

        with torch.no_grad():
            source_idx = station_to_idx[source_station]

            for target_station in target_stations:
                if target_station in station_to_idx:
                    target_idx = station_to_idx[target_station]

                    # Predict flow
                    time_vec = torch.tensor([[hour/23.0, day_of_week/6.0]], dtype=torch.float32).to(self.device)
                    prediction = self.model(
                        self.graph_data.x.to(self.device),
                        self.graph_data.edge_index.to(self.device),
                        self.graph_data.edge_attr.to(self.device),
                        torch.tensor([source_idx]).to(self.device),
                        torch.tensor([target_idx]).to(self.device),
                        time_vec
                    )
                    # invert log transform if used
                    pred_val = prediction.item()
                    if getattr(self.config, 'log_labels', False):
                        pred_val = np.expm1(pred_val)
                    predictions[target_station] = max(0.0, pred_val)

        return predictions
    
    def predict_flow(self, source_station, target_station, hour, day_of_week):
        """Predict flow between two specific stations using trained GNN"""
        result = self.predict_flows_gnn(source_station, [target_station], hour, day_of_week)
        predicted_flow = result.get(target_station, 0.0)
        
        # Simple confidence calculation based on flow magnitude
        confidence = min(1.0, predicted_flow / 10.0)
        
        return predicted_flow, confidence
    
    def _extract_basic_features_only(self):
        """Extract only basic features for quick testing"""
        logger.info("Extracting basic features only...")
        
        for station_id, (lat, lon) in self.station_coords.items():
            # Basic station features
            station_trips = self.trips_df[
                (self.trips_df['start_station_id'] == station_id) | 
                (self.trips_df['end_station_id'] == station_id)
            ]
            
            features = {
                'latitude': lat,
                'longitude': lon,
                'total_trips': len(station_trips),
                'avg_temperature': station_trips.get('temperature', pd.Series([20.0])).mean(),
            }
            
            # Add basic temporal features
            for hour in [8, 12, 17]:
                hour_trips = len(station_trips[station_trips.get('hour', pd.Series([])) == hour])
                features[f'trips_hour_{hour}'] = hour_trips
            
            self.station_features[station_id] = features
    
    def _prepare_minimal_training_data(self, max_samples: int = 100, offset: int = 0):
        """Prepare minimal training data for quick testing"""
        if not self.station_features:
            return []
        
        training_samples = []
        station_ids = list(self.station_coords.keys())
        station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
        
        # Create simple flow data
        for i, source_id in enumerate(station_ids[:10]):  # Limit to first 10 stations
            for j, target_id in enumerate(station_ids[:10]):
                if source_id != target_id and len(training_samples) < max_samples + offset:
                    
                    # Skip offset samples
                    if len(training_samples) < offset:
                        continue
                    
                    # Create synthetic flow based on distance and features
                    source_features = self.station_features[source_id]
                    target_features = self.station_features[target_id]
                    
                    # Simple distance-based flow
                    lat_diff = abs(source_features['latitude'] - target_features['latitude'])
                    lon_diff = abs(source_features['longitude'] - target_features['longitude'])
                    distance = np.sqrt(lat_diff**2 + lon_diff**2)
                    
                    # Synthetic flow (inverse distance with noise)
                    flow = max(1, 10 / (1 + distance * 100)) + np.random.normal(0, 0.5)
                    flow = max(0, flow)
                    
                    training_samples.append({
                        'source_idx': station_to_idx[source_id],  # Fixed: use correct key names
                        'target_idx': station_to_idx[target_id],  # Fixed: use correct key names
                        'source_station': source_id,
                        'target_station': target_id,
                        'hour': 12,  # Fixed hour for simplicity
                        'day_of_week': 1,  # Fixed day
                        'flow': flow
                    })
        
        logger.info(f"Created {len(training_samples)} minimal training samples")
        return training_samples
    
    def _evaluate_on_samples(self, test_samples):
        """Evaluate model on test samples"""
        if not test_samples or self.model is None:
            return {'mse': float('inf'), 'mae': float('inf'), 'r2': -float('inf'), 'converged': False}
        
        predictions = []
        actuals = []
        
        for sample in test_samples:
            try:
                pred_flow, _ = self.predict_flow(
                    sample['source_station'], 
                    sample['target_station'], 
                    sample['hour'], 
                    sample['day_of_week']
                )
                predictions.append(pred_flow)
                actuals.append(sample['flow'])
            except:
                continue
        
        if len(predictions) == 0:
            return {'mse': float('inf'), 'mae': float('inf'), 'r2': -float('inf'), 'converged': False}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        try:
            r2 = r2_score(actuals, predictions)
        except:
            r2 = -float('inf')
        
        # Check convergence (simple heuristic)
        converged = mae < 10.0 and not np.isnan(mae) and not np.isinf(mae)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'converged': converged
        }
    
    def get_model_summary(self):
        """Get model summary for evaluation"""
        if self.model is None:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'architecture': self.config.gnn_type,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'parameters': sum(p.numel() for p in self.model.parameters())
        }
    
    def get_model_summary(self):
        """Get summary of the GNN model"""
        if self.model is None:
            return "Model not initialized"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'config': self.config,
            'graph_nodes': self.graph_data.x.shape[0] if self.graph_data else 0,
            'graph_edges': self.graph_data.edge_index.shape[1] if self.graph_data else 0,
            'node_features': self.graph_data.x.shape[1] if self.graph_data else 0
        }
