#!/usr/bin/env python3
"""
Train and save models for the Flow Prediction Lab
"""
import pickle
import os
from datetime import datetime
from helpers.run_gnn_baselines import GNNBaselineRunner

# Add DCRNN support
from add_dcrnn_methods import *

def train_and_save_models():
    """Train models and save to pickle files"""
    print("🚀 Starting model training...")
    
    # Initialize runner
    runner = GNNBaselineRunner()
    success = runner.load_data()
    
    if not success:
        print("❌ Failed to load data")
        return
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train lightweight models including DCRNN
    models_to_train = [
        ("xgboost", lambda: runner.run_xgboost_baseline(500)),
        ("lightgbm", lambda: runner.run_lightgbm_baseline(500)),
        ("rf", lambda: runner.run_rf_baseline(500)),
        ("dcrnn", lambda: runner.run_dcrnn_baseline(500)),
        ("stgcn", lambda: runner.run_stgcn_baseline(500))
    ]
    
    trained_models = {}
    
    for model_name, train_func in models_to_train:
        print(f"🔄 Training {model_name}...")
        try:
            result = train_func()
            if result and 'rmse' in result:
                # Save model result
                model_data = {
                    'result': result,
                    'runner': runner,
                    'trained_at': datetime.now().isoformat(),
                    'model_name': model_name
                }
                
                # Save to pickle
                model_path = f"models/{model_name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                trained_models[model_name] = result
                print(f"✅ {model_name}: RMSE={result.get('rmse', 0):.3f}, R²={result.get('r2', 0):.3f}")
            else:
                print(f"❌ {model_name} training failed")
                
        except Exception as e:
            print(f"❌ {model_name} error: {e}")
    
    # Save summary
    summary = {
        'trained_models': trained_models,
        'training_time': datetime.now().isoformat(),
        'data_info': {
            'stations': len(runner.osm_features[500]['station_id'].unique()) if hasattr(runner, 'osm_features') else 0,
            'features': list(runner.osm_features[500].columns) if hasattr(runner, 'osm_features') else []
        }
    }
    
    with open("models/training_summary.pkl", 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"🎉 Training complete! {len(trained_models)} models saved to models/")
    return trained_models

if __name__ == "__main__":
    train_and_save_models()