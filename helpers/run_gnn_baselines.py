#!/usr/bin/env python3
"""
GNN Baseline Testing with OSM Features
=====================================

Test GCN and GraphSAGE models using the extracted OSM features.
"""

import pandas as pd
import numpy as np
import torch
import logging
import sys
import os
from typing import Dict, List

# Import GNN components
from gnn_flow_predictor import GNNFlowPredictor, GNNConfig
from gnn_testing_framework import GNNTester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GNNBaselineRunner:
    def run_xgboost_baseline(self, radius):
        """Run XGBoost regression using OSM features and aggregated flows."""
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except ImportError:
            logger.error("XGBoost not installed. Please install with: pip install xgboost")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return None
        # Merge OSM features for start and end stations
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        # Merge start features
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        # Merge end features
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        # Drop non-feature columns
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        # Hyperparameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
        }
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"XGBoost (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': best_model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius}
    def prepare_basic_flow_samples(self, radius):
        """Prepare training samples from aggregated flows (no time granularity)."""
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return []
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return []
        osm_features_df = self.osm_features[radius]
        station_ids = list(osm_features_df['station_id'])
        station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
        samples = []
        for _, row in self.flow_df.iterrows():
            s = row['start_station_id']
            t = row['end_station_id']
            flow = row['flow']
            if s in station_to_idx and t in station_to_idx:
                samples.append({
                    'source_idx': station_to_idx[s],
                    'target_idx': station_to_idx[t],
                    'flow': flow,
                    'time_vec': [0.5, 0.5]  # dummy if not using time
                })
        logger.info(f"Prepared {len(samples)} basic flow samples for radius {radius}")
        return samples
    def aggregate_flows(self, by_time=False):
        """Aggregate trips to compute flows between station pairs. Optionally by hour/day."""
        if self.trips_df is None:
            logger.error("Trip data not loaded.")
            return None
        df = self.trips_df.copy()
        # Parse time columns if needed
        if by_time:
            df['start_time_dt'] = pd.to_datetime(df['start_time'], format='%Y%m%d_%H%M%S', errors='coerce')
            df['hour'] = df['start_time_dt'].dt.hour
            df['dayofweek'] = df['start_time_dt'].dt.dayofweek
            group_cols = ['start_station_id', 'end_station_id', 'hour', 'dayofweek']
        else:
            group_cols = ['start_station_id', 'end_station_id']
        # Count trips (flows)
        flow_df = df.groupby(group_cols).size().reset_index(name='flow')
        logger.info(f"Aggregated flows: {len(flow_df)} rows (by_time={by_time})")
        return flow_df
    def check_data_quality(self):
        """Analyze data quality: missing values, variance, outliers, and feature-target correlation."""
        logger.info("\n===== DATA QUALITY CHECK =====")
        if self.trips_df is None:
            logger.error("Trip data not loaded.")
            return
        df = self.trips_df
        # Missing values
        missing = df.isnull().sum()
        logger.info(f"Missing values per column:\n{missing[missing > 0] if missing.sum() > 0 else 'None'}")
        # Low variance features
        numeric = df.select_dtypes(include=[np.number])
        low_var = numeric.var()[numeric.var() < 1e-5]
        logger.info(f"Low variance features:\n{low_var if not low_var.empty else 'None'}")
        # Outliers (z-score > 4)
        zscores = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0))
        outlier_counts = (zscores > 4).sum()
        logger.info(f"Outlier counts per feature (z-score > 4):\n{outlier_counts[outlier_counts > 0] if outlier_counts.sum() > 0 else 'None'}")
        # Feature-target correlation
        target_col = None
        for col in ['flow', 'target', 'label', 'y']:
            if col in numeric.columns:
                target_col = col
                break
        if target_col:
            corrs = numeric.corr()[target_col].drop(target_col)
            logger.info(f"Feature-target correlations (Pearson):\n{corrs}")
        else:
            logger.warning("No target column found for correlation analysis.")
        logger.info("===== END DATA QUALITY CHECK =====\n")
    """Run GNN baselines with OSM features"""
    
    def __init__(self):
        self.trips_df = None
        self.osm_features = {}
        self.results = {}
        
    def load_data(self):
        """Load trip data and OSM features"""
        logger.info("Loading datasets for GNN testing...")
        
        # Load trips
        self.trips_df = pd.read_csv('data/trips_8days_flat.csv')
        logger.info(f"Loaded {len(self.trips_df)} trip records")
        
        # Load OSM features for different radii
        radii = [500, 1000, 1500]
        for radius in radii:
            try:
                osm_file = f'cache/ml_ready/switzerland_station_features_{radius}m.csv'
                osm_df = pd.read_csv(osm_file)
                self.osm_features[radius] = osm_df
                logger.info(f"Loaded OSM features for {radius}m radius: {len(osm_df)} stations")
            except FileNotFoundError:
                logger.warning(f"OSM features file not found for {radius}m radius")
        
        # Aggregate flows and store as attribute for use as target
        self.flow_df = self.aggregate_flows(by_time=False)
        return True
    
    def create_gnn_configs(self, device: torch.device = None) -> List[GNNConfig]:
        """Create different GNN configurations to test"""
        configs = []
        
        # Adjust batch size based on device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use larger batch sizes for GPU
        batch_size = 64 if device.type == 'cuda' else 32
        epochs = 150 if device.type == 'cuda' else 100  # More epochs on GPU
        
        # GCN Baseline - Simple and robust
        gcn_config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.01,
            batch_size=batch_size,
            epochs=epochs,
            gnn_type="GCN",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('GCN_Baseline', gcn_config))
        
        # GraphSAGE - Good for larger graphs
        sage_config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.01,
            batch_size=batch_size,
            epochs=epochs,
            gnn_type="GraphSAGE",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('GraphSAGE_Baseline', sage_config))
        
        # GAT - Attention-based (if you want to test)
        gat_config = GNNConfig(
            hidden_dim=32,  # Smaller due to multi-head attention
            num_layers=2,
            dropout=0.3,
            learning_rate=0.01,
            batch_size=batch_size // 2,  # Smaller batch for GAT due to memory usage
            epochs=epochs,
            attention_heads=4,
            gnn_type="GAT",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('GAT_Baseline', gat_config))
        
        return configs
    
    def run_gnn_baseline(self, config_name: str, config: GNNConfig, radius: int, device: torch.device = None) -> Dict:
        """Run a single GNN configuration"""
        logger.info(f"\\n{'='*60}")
        logger.info(f"TESTING {config_name} with {radius}m OSM features")
        logger.info(f"{'='*60}")
        
        try:
            # Set default device if not provided
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            # Create GNN predictor and instruct it to load the explicit per-radius cache file
            cache_file = f'cache/ml_ready/switzerland_station_features_{radius}m.csv'
            gnn_predictor = GNNFlowPredictor(self.trips_df, config, use_cached_features=True, cache_file=cache_file)
            
            # Set device on the predictor
            gnn_predictor.device = device
            
            # Add OSM features if available
            if radius in self.osm_features:
                logger.info(f"Adding OSM features from {radius}m radius...")
                
                # Get OSM features and directly set them on the predictor
                osm_features_df = self.osm_features[radius]
                
                # Extract station coordinates from OSM features
                logger.info(f"Building station network from OSM features...")
                gnn_predictor.station_coords = {}
                gnn_predictor.station_features = {}
                
                for _, row in osm_features_df.iterrows():
                    station_id = row['station_id']
                    
                    # Extract coordinates
                    gnn_predictor.station_coords[station_id] = (row['lat'], row['lon'])
                    
                    # Extract all feature columns (excluding station_id, lat, lon, coords)
                    features = {}
                    for col in osm_features_df.columns:
                        if col not in ['station_id', 'lat', 'lon', 'coords']:
                            try:
                                # Only include numeric features
                                val = float(row[col])
                                features[col] = val
                            except (ValueError, TypeError):
                                # Skip non-numeric columns
                                continue
                    
                    gnn_predictor.station_features[station_id] = features
                
                logger.info(f"Loaded {len(gnn_predictor.station_coords)} stations with {len(features)} features each")
                
                # Build graph structure directly
                logger.info("Building graph structure...")
                station_to_idx, feature_names = gnn_predictor.build_graph_structure()
                
                # Prepare training data
                logger.info("Preparing training data...")
                # Use aggregated flows for basic baseline, else fallback to time-aware
                if hasattr(self, 'flow_df') and self.flow_df is not None:
                    training_samples = self.prepare_basic_flow_samples(radius)
                else:
                    training_samples = gnn_predictor.prepare_training_data()

                if len(training_samples) == 0:
                    logger.warning("No training samples generated")
                    return None
                
                # Train the model
                logger.info(f"Training GNN model with {len(training_samples)} samples...")
                gnn_predictor.train_gnn_model(training_samples)
                
                # Basic evaluation - predict some flows and calculate metrics
                logger.info("Evaluating model...")
                test_metrics = self.evaluate_gnn_model(gnn_predictor, training_samples[:100])
                
                # Combine results
                results = {
                    'config_name': config_name,
                    'radius': radius,
                    'gnn_type': config.gnn_type,
                    'training_samples': len(training_samples),
                    'test_rmse': test_metrics.get('rmse', 0),
                    'test_mae': test_metrics.get('mae', 0),
                    'test_r2': test_metrics.get('r2', 0),
                    'num_stations': len(station_to_idx),
                    'num_features': len(feature_names),
                    'config': config.__dict__
                }
                
                logger.info(f"Results: RMSE={results['test_rmse']:.2f}, MAE={results['test_mae']:.2f}, R²={results['test_r2']:.3f}")
                return results
                
            else:
                logger.error(f"No OSM features available for {radius}m radius")
                return None
                
        except Exception as e:
            logger.error(f"Error running {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_gnn_model(self, gnn_predictor, test_samples):
        """Evaluate GNN model on test samples"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        predictions = []
        actuals = []
        
        gnn_predictor.model.eval()
        
        with torch.no_grad():
            # Get graph data
            graph_data = gnn_predictor.graph_data.to(gnn_predictor.device)

            # Create batch from test samples
            source_indices = torch.tensor([sample['source_idx'] for sample in test_samples], dtype=torch.long).to(gnn_predictor.device)
            target_indices = torch.tensor([sample['target_idx'] for sample in test_samples], dtype=torch.long).to(gnn_predictor.device)
            flows = torch.tensor([sample['flow'] for sample in test_samples], dtype=torch.float32)
            # Prepare time features if present
            time_feats = torch.tensor([sample.get('time_vec', [0.5, 0.5]) for sample in test_samples], dtype=torch.float32).to(gnn_predictor.device)

            # Get predictions
            pred_flows = gnn_predictor.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                source_indices,
                target_indices,
                time_feats
            )

            predictions = pred_flows.cpu().numpy().flatten()
            actuals = flows.numpy().flatten()

            # Clear GPU memory
            if gnn_predictor.device.type == 'cuda':
                del graph_data, source_indices, target_indices, pred_flows, time_feats
                torch.cuda.empty_cache()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def run_all_baselines(self, device: torch.device = None):
        """Run all GNN baselines with different OSM radii"""
        logger.info("🚀 Starting GNN Baseline Testing with OSM Features")
        
        # Set default device if not provided
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {device}")
        
        if not self.load_data():
            logger.error("Failed to load data")
            return
        self.check_data_quality()
        
        configs = self.create_gnn_configs(device)
        radii = [500, 1000, 1500]  # Test all radii
        
        all_results = []
        
        # Test all configurations
        for config_name, config in configs:
            for radius in radii:
                if radius in self.osm_features:
                    result = self.run_gnn_baseline(config_name, config, radius, device)
                    if result:
                        all_results.append(result)
                        
                        # Store in class results
                        key = f"{config_name}_{radius}m"
                        self.results[key] = result
                        
                        # Clear GPU cache after each run to prevent memory issues
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
        
        # Run XGBoost baselines for each radius
        for radius in radii:
            if radius in self.osm_features:
                xgb_result = self.run_xgboost_baseline(radius)
                if xgb_result:
                    all_results.append({
                        'config_name': 'XGBoost_Baseline',
                        'radius': radius,
                        'gnn_type': 'XGBoost',
                        'training_samples': None,
                        'test_rmse': xgb_result['rmse'],
                        'test_mae': xgb_result['mae'],
                        'test_r2': xgb_result['r2'],
                        'num_stations': None,
                        'num_features': None,
                        'config': None
                    })

        # Save results
        self.save_results(all_results)
        self.print_summary(all_results)
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Save results to CSV"""
        if not results:
            logger.warning("No results to save")
            return
            
        results_df = pd.DataFrame(results)
        results_file = f"results/gnn_baseline_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        os.makedirs('results', exist_ok=True)
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary of results"""
        if not results:
            return
            
        logger.info("\\n" + "="*80)
        logger.info("GNN BASELINE RESULTS SUMMARY")
        logger.info("="*80)
        
        df = pd.DataFrame(results)
        
        # Group by GNN type and radius
        for gnn_type in df['gnn_type'].unique():
            logger.info(f"\\n📊 {gnn_type} Results:")
            gnn_results = df[df['gnn_type'] == gnn_type]
            
            for _, row in gnn_results.iterrows():
                logger.info(f"  {row['radius']}m: RMSE={row['test_rmse']:.2f}, MAE={row['test_mae']:.2f}, R²={row['test_r2']:.3f}")
        
        # Best results
        best_result = df.loc[df['test_r2'].idxmax()]
        logger.info(f"\\n🏆 BEST RESULT:")
        logger.info(f"  Model: {best_result['config_name']} ({best_result['radius']}m)")
        logger.info(f"  RMSE: {best_result['test_rmse']:.2f}")
        logger.info(f"  MAE: {best_result['test_mae']:.2f}")
        logger.info(f"  R²: {best_result['test_r2']:.3f}")

def main():
    """Main execution"""
    # Check if PyTorch Geometric is available
    try:
        import torch_geometric
        logger.info(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        logger.error("PyTorch Geometric not installed. Please install with: pip install torch-geometric")
        return
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run baselines
    runner = GNNBaselineRunner()
    runner.run_all_baselines(device)
    
    logger.info("🎉 GNN baseline testing completed!")

if __name__ == "__main__":
    main()
