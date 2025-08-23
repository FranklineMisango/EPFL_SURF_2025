#!/usr/bin/env python3
"""
Model Loader Utility for Flow Prediction Lab
Handles loading and using trained models saved with pickle
"""

import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Utility class for loading and using trained flow prediction models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_metadata = {}
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available trained models"""
        if not os.path.exists(self.models_dir):
            return []
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        available_models = []
        
        for model_file in model_files:
            try:
                model_path = os.path.join(self.models_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                model_info = {
                    'name': model_data.get('model_name', 'Unknown'),
                    'file': model_file,
                    'path': model_path,
                    'metrics': model_data.get('metrics', {}),
                    'trained_at': model_data.get('trained_at', 'Unknown'),
                    'radius': model_data.get('radius', 500),
                    'size_mb': os.path.getsize(model_path) / (1024 * 1024)
                }
                
                available_models.append(model_info)
                
            except Exception as e:
                logger.warning(f"Failed to read model metadata from {model_file}: {e}")
                continue
        
        # Sort by training date (newest first)
        available_models.sort(key=lambda x: x['trained_at'], reverse=True)
        return available_models
    
    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load a specific model by name"""
        # Check if already loaded
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Find model file
        model_file = f"{model_name.lower()}_model.pkl"
        model_path = os.path.join(self.models_dir, model_file)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.loaded_models[model_name] = model_data
            self.model_metadata[model_name] = {
                'loaded_at': datetime.now().isoformat(),
                'file_path': model_path,
                'metrics': model_data.get('metrics', {}),
                'radius': model_data.get('radius', 500)
            }
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def predict_flow(self, model_name: str, source_station: str, target_station: str, 
                    hour: int = 12, day_of_week: int = 1, **kwargs) -> Dict[str, Any]:
        """Predict flow using a loaded model"""
        model_data = self.load_model(model_name)
        
        if model_data is None:
            return {'error': f'Model {model_name} not found or failed to load'}
        
        try:
            # Get the trained model and runner state
            result = model_data.get('result', {})
            runner_state = model_data.get('runner_state', {})
            trained_model = model_data.get('trained_model')
            
            # For tree-based models (XGBoost, LightGBM, etc.)
            if trained_model and hasattr(trained_model, 'predict'):
                return self._predict_with_sklearn_model(
                    trained_model, source_station, target_station, 
                    hour, day_of_week, runner_state, **kwargs
                )
            
            # For GNN models
            elif 'gnn_type' in result:
                return self._predict_with_gnn_model(
                    model_data, source_station, target_station,
                    hour, day_of_week, **kwargs
                )
            
            # Fallback: return model metadata for inspection
            else:
                return {
                    'prediction': 0.0,
                    'confidence': 0.0,
                    'model_type': model_data.get('model_name', 'Unknown'),
                    'warning': 'Model prediction not implemented for this type',
                    'available_data': list(model_data.keys())
                }
                
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _predict_with_sklearn_model(self, model, source_station: str, target_station: str,
                                   hour: int, day_of_week: int, runner_state: Dict, **kwargs) -> Dict[str, Any]:
        """Predict using sklearn-compatible models"""
        try:
            # Get station features from runner state
            osm_features = runner_state.get('osm_features', {})
            
            if not osm_features:
                return {'error': 'No OSM features available for prediction'}
            
            # Get features for source and target stations
            source_features = osm_features.get(500, {})  # Default to 500m radius
            
            # This is a simplified prediction - in practice, you'd need to reconstruct
            # the exact feature vector used during training
            if isinstance(source_features, dict) and source_station in source_features:
                # Create a simple feature vector (this would need to match training exactly)
                features = np.array([[hour/24.0, day_of_week/7.0, 0.5, 0.5]])  # Simplified
                
                prediction = model.predict(features)[0]
                confidence = min(1.0, prediction / 10.0)  # Simple confidence estimate
                
                return {
                    'prediction': max(0.0, float(prediction)),
                    'confidence': float(confidence),
                    'model_type': 'sklearn',
                    'features_used': features.shape[1]
                }
            else:
                return {'error': f'Station {source_station} not found in training data'}
                
        except Exception as e:
            return {'error': f'Sklearn prediction failed: {str(e)}'}
    
    def _predict_with_gnn_model(self, model_data: Dict, source_station: str, target_station: str,
                               hour: int, day_of_week: int, **kwargs) -> Dict[str, Any]:
        """Predict using GNN models"""
        try:
            # GNN prediction would require reconstructing the graph and running inference
            # This is a placeholder implementation
            
            result = model_data.get('result', {})
            model_type = result.get('gnn_type', 'Unknown')
            
            # Simple heuristic prediction based on model performance
            base_prediction = 5.0  # Base flow
            rmse = result.get('rmse', 10.0)
            r2 = result.get('r2', 0.0)
            
            # Adjust prediction based on time
            time_factor = 1.0
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                time_factor = 1.5
            elif 22 <= hour or hour <= 6:  # Night hours
                time_factor = 0.3
            
            prediction = base_prediction * time_factor
            confidence = max(0.1, min(1.0, r2)) if r2 > 0 else 0.1
            
            return {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'model_type': f'GNN_{model_type}',
                'note': 'Simplified GNN prediction - full implementation requires graph reconstruction'
            }
            
        except Exception as e:
            return {'error': f'GNN prediction failed: {str(e)}'}
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        model_data = self.load_model(model_name)
        
        if model_data is None:
            return None
        
        info = {
            'name': model_data.get('model_name', model_name),
            'trained_at': model_data.get('trained_at', 'Unknown'),
            'radius': model_data.get('radius', 500),
            'metrics': model_data.get('metrics', {}),
            'model_type': 'Unknown'
        }
        
        # Determine model type
        result = model_data.get('result', {})
        if 'gnn_type' in result:
            info['model_type'] = f"GNN_{result['gnn_type']}"
        elif 'trained_model' in model_data:
            model_obj = model_data['trained_model']
            info['model_type'] = type(model_obj).__name__
        
        # Add runner state info
        runner_state = model_data.get('runner_state', {})
        if runner_state:
            info['stations_count'] = len(runner_state.get('station_coords', {}))
            info['has_osm_features'] = bool(runner_state.get('osm_features'))
            info['has_flow_data'] = bool(runner_state.get('flow_df'))
        
        return info
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare multiple models side by side"""
        comparison_data = []
        
        for model_name in model_names:
            info = self.get_model_info(model_name)
            if info:
                metrics = info.get('metrics', {})
                comparison_data.append({
                    'Model': info['name'],
                    'Type': info['model_type'],
                    'RMSE': metrics.get('rmse', 'N/A'),
                    'MAE': metrics.get('mae', 'N/A'),
                    'R²': metrics.get('r2', 'N/A'),
                    'Accuracy %': metrics.get('accuracy_pct', 'N/A'),
                    'Radius (m)': info['radius'],
                    'Trained': info['trained_at'][:10] if info['trained_at'] != 'Unknown' else 'Unknown'
                })
        
        return pd.DataFrame(comparison_data)
    
    def cleanup_old_models(self, keep_latest: int = 5):
        """Remove old model files, keeping only the latest N models"""
        available_models = self.list_available_models()
        
        if len(available_models) <= keep_latest:
            logger.info(f"Only {len(available_models)} models found, no cleanup needed")
            return
        
        # Sort by training date and remove oldest
        models_to_remove = available_models[keep_latest:]
        
        for model_info in models_to_remove:
            try:
                os.remove(model_info['path'])
                logger.info(f"Removed old model: {model_info['file']}")
            except Exception as e:
                logger.warning(f"Failed to remove {model_info['file']}: {e}")
        
        logger.info(f"Cleanup complete: removed {len(models_to_remove)} old models")

# Convenience functions for easy usage
def load_model(model_name: str, models_dir: str = "models") -> Optional[Dict[str, Any]]:
    """Quick function to load a model"""
    loader = ModelLoader(models_dir)
    return loader.load_model(model_name)

def predict_flow(model_name: str, source_station: str, target_station: str,
                hour: int = 12, day_of_week: int = 1, models_dir: str = "models") -> Dict[str, Any]:
    """Quick function to predict flow"""
    loader = ModelLoader(models_dir)
    return loader.predict_flow(model_name, source_station, target_station, hour, day_of_week)

def list_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """Quick function to list available models"""
    loader = ModelLoader(models_dir)
    return loader.list_available_models()

if __name__ == "__main__":
    # Demo usage
    loader = ModelLoader()
    
    print("📊 Available Models:")
    models = loader.list_available_models()
    
    if not models:
        print("No trained models found. Run training first.")
    else:
        for model in models:
            print(f"  • {model['name']} ({model['file']})")
            print(f"    RMSE: {model['metrics'].get('rmse', 'N/A')}")
            print(f"    R²: {model['metrics'].get('r2', 'N/A')}")
            print(f"    Size: {model['size_mb']:.1f} MB")
            print()
