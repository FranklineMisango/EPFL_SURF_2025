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
        
        return True
    
    def create_gnn_configs(self) -> List[GNNConfig]:
        """Create different GNN configurations to test"""
        configs = []
        
        # GCN Baseline - Simple and robust
        gcn_config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.01,
            batch_size=32,
                epochs=100,  # Increased to improve training
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
            batch_size=32,
                epochs=100,  # Increased to improve training
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
            batch_size=32,
                epochs=100,  # Increased to improve training
            attention_heads=4,
            gnn_type="GAT",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('GAT_Baseline', gat_config))
        
        return configs
    
    def run_gnn_baseline(self, config_name: str, config: GNNConfig, radius: int) -> Dict:
        """Run a single GNN configuration"""
        logger.info(f"\\n{'='*60}")
        logger.info(f"TESTING {config_name} with {radius}m OSM features")
        logger.info(f"{'='*60}")
        
        try:
            # Create GNN predictor and instruct it to load the explicit per-radius cache file
            cache_file = f'cache/ml_ready/switzerland_station_features_{radius}m.csv'
            gnn_predictor = GNNFlowPredictor(self.trips_df, config, use_cached_features=True, cache_file=cache_file)
            
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
            
            # Get predictions
            pred_flows = gnn_predictor.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                source_indices,
                target_indices
            )
            
            predictions = pred_flows.cpu().numpy().flatten()
            actuals = flows.numpy().flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def run_all_baselines(self):
        """Run all GNN baselines with different OSM radii"""
        logger.info("🚀 Starting GNN Baseline Testing with OSM Features")
        
        if not self.load_data():
            logger.error("Failed to load data")
            return
        
        configs = self.create_gnn_configs()
        radii = [500, 1000, 1500]  # Test all radii
        
        all_results = []
        
        # Test all configurations
        for config_name, config in configs:
            for radius in radii:
                if radius in self.osm_features:
                    result = self.run_gnn_baseline(config_name, config, radius)
                    if result:
                        all_results.append(result)
                        
                        # Store in class results
                        key = f"{config_name}_{radius}m"
                        self.results[key] = result
        
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
    
    # Run baselines
    runner = GNNBaselineRunner()
    runner.run_all_baselines()
    
    logger.info("🎉 GNN baseline testing completed!")

if __name__ == "__main__":
    main()
