#!/usr/bin/env python3
"""
Train and save individual models for the Flow Prediction Lab
Supports all 16 model types with pickle serialization
"""

import pickle
import os
import sys
import argparse
import torch
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from helpers.run_gnn_baselines import GNNBaselineRunner, GNNConfig

def get_available_models():
    """Return list of all available models"""
    return [
        'DCRNN', 'GCN', 'GraphSAGE', 'GAT', 'GIN', 'Transformer',
        'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'MLP', 'TabNet',
        'Stacking', 'Blending', 'ST-GCN', 'TemporalFusionTransformer'
    ]

def train_gnn_model(runner, model_name, radius=500):
    """Train a specific GNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = runner.create_gnn_configs(device)
    
    # Find matching config
    for config_name, config in configs:
        if model_name.lower() in config_name.lower():
            print(f"🔄 Training {model_name} with {radius}m radius...")
            result = runner.run_gnn_baseline(config_name, config, radius, device)
            if result:
                # Convert GNN result format to match other models
                return {
                    'model': None,  # GNN models don't return the actual model object
                    'rmse': result.get('test_rmse'),
                    'mae': result.get('test_mae'), 
                    'r2': result.get('test_r2'),
                    'radius': radius
                }
            return result
    
    return None

def train_single_model(model_name, radius=500, save_path=None):
    """Train and save a single model"""
    print(f"🚀 Starting {model_name} model training...")
    
    # Initialize runner
    runner = GNNBaselineRunner()
    success = runner.load_data()
    
    if not success:
        print("❌ Failed to load data")
        return None
    
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    result = None
    model_data = None
    
    try:
        # Train based on model type
        if model_name.upper() == 'DCRNN':
            result = train_gnn_model(runner, 'DCRNN', radius)
        elif model_name.upper() == 'GCN':
            result = train_gnn_model(runner, 'GCN', radius)
        elif model_name.upper() == 'GRAPHSAGE':
            result = train_gnn_model(runner, 'GraphSAGE', radius)
        elif model_name.upper() == 'GAT':
            result = train_gnn_model(runner, 'GAT', radius)
        elif model_name.upper() == 'GIN':
            result = train_gnn_model(runner, 'GIN', radius)
        elif model_name.upper() == 'TRANSFORMER':
            result = train_gnn_model(runner, 'Transformer', radius)
        elif model_name.upper() == 'XGBOOST':
            result = runner.run_xgboost_baseline(radius)
        elif model_name.upper() == 'LIGHTGBM':
            result = runner.run_lightgbm_baseline(radius)
        elif model_name.upper() == 'CATBOOST':
            result = runner.run_catboost_baseline(radius)
        elif model_name.upper() == 'RANDOMFOREST':
            result = runner.run_rf_baseline(radius)
        elif model_name.upper() == 'MLP':
            result = runner.run_mlp_baseline(radius)
        elif model_name.upper() == 'TABNET':
            result = runner.run_tabnet_baseline(radius)
        elif model_name.upper() == 'STACKING':
            result = runner.run_stacking_ensemble(radius)
        elif model_name.upper() == 'BLENDING':
            result = runner.run_blending_ensemble(radius)
        elif model_name.upper() == 'ST-GCN':
            result = runner.run_stgcn_baseline(radius)
        elif model_name.upper() == 'TEMPORALFUSIONTRANSFORMER':
            result = runner.run_temporal_fusion_transformer(radius)
        else:
            print(f"❌ Unknown model: {model_name}")
            return None
        
        if result is None:
            print(f"❌ {model_name} training failed - no result returned")
            return None
        
        # Prepare model data for saving
        model_data = {
            'model_name': model_name,
            'result': result,
            'radius': radius,
            'trained_at': datetime.now().isoformat(),
            'runner_state': {
                'station_coords': getattr(runner, 'station_coords', None),
                'osm_features': runner.osm_features,
                'flow_df': runner.flow_df.to_dict() if hasattr(runner, 'flow_df') and runner.flow_df is not None else None
            },
            'metrics': {
                'rmse': result.get('rmse', None),
                'mae': result.get('mae', None),
                'r2': result.get('r2', None),
                'accuracy_pct': result.get('accuracy_pct', None)
            }
        }
        
        # Add model object if available
        if 'model' in result:
            model_data['trained_model'] = result['model']
        
        # Determine save path
        if save_path is None:
            save_path = os.path.join(models_dir, f"{model_name.lower()}_model.pkl")
        
        # Save model
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ {model_name} training completed!")
        rmse = result.get('rmse', 'N/A')
        mae = result.get('mae', 'N/A')
        r2 = result.get('r2', 'N/A')
        rmse_str = f"{rmse:.3f}" if isinstance(rmse, (int, float)) else str(rmse)
        mae_str = f"{mae:.3f}" if isinstance(mae, (int, float)) else str(mae)
        r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else str(r2)
        print(f"📊 Metrics: RMSE={rmse_str}, MAE={mae_str}, R²={r2_str}")
        if 'accuracy_pct' in result:
            acc = result.get('accuracy_pct', 'N/A')
            acc_str = f"{acc:.1f}%" if isinstance(acc, (int, float)) else str(acc)
            print(f"🎯 Accuracy: {acc_str}")
        print(f"💾 Model saved to: {save_path}")
        
        return model_data
        
    except Exception as e:
        print(f"❌ {model_name} training error: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_trained_model(model_path):
    """Load a trained model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}: {e}")
        return None

def list_trained_models():
    """List all trained models in the models directory"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("📁 No models directory found")
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("📁 No trained models found")
        return []
    
    print("📊 Trained models:")
    trained_models = []
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_data = load_trained_model(model_path)
        
        if model_data:
            model_name = model_data.get('model_name', 'Unknown')
            metrics = model_data.get('metrics', {})
            trained_at = model_data.get('trained_at', 'Unknown')
            
            print(f"  • {model_name}")
            print(f"    File: {model_file}")
            print(f"    RMSE: {metrics.get('rmse', 'N/A')}")
            print(f"    R²: {metrics.get('r2', 'N/A')}")
            print(f"    Trained: {trained_at}")
            print()
            
            trained_models.append({
                'name': model_name,
                'file': model_file,
                'path': model_path,
                'metrics': metrics,
                'trained_at': trained_at
            })
    
    return trained_models

def interactive_menu():
    """Interactive menu for model training"""
    models = get_available_models()
    
    while True:
        print("\n" + "="*60)
        print("🤖 FLOW PREDICTION MODEL TRAINER")
        print("="*60)
        print("\nSelect a model to train:")
        
        for i, model in enumerate(models, 1):
            print(f"  {i:2d}. {model}")
        
        print(f"  {len(models)+1:2d}. List trained models")
        print(f"  {len(models)+2:2d}. Exit")
        
        try:
            choice = input(f"\nEnter choice (1-{len(models)+2}): ").strip()
            
            if choice == str(len(models)+2):  # Exit
                print("👋 Goodbye!")
                break
            elif choice == str(len(models)+1):  # List models
                list_trained_models()
                continue
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected_model = models[choice_idx]
                
                # Ask for radius
                radius_input = input(f"Enter radius in meters (default: 500): ").strip()
                radius = int(radius_input) if radius_input.isdigit() else 500
                
                # Train the model
                result = train_single_model(selected_model, radius)
                
                if result:
                    print(f"\n🎉 {selected_model} training completed successfully!")
                else:
                    print(f"\n❌ {selected_model} training failed!")
                
                input("\nPress Enter to continue...")
            else:
                print("❌ Invalid choice!")
                
        except ValueError:
            print("❌ Please enter a valid number!")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train and save individual flow prediction models")
    parser.add_argument('--model', type=str, help='Model name to train')
    parser.add_argument('--radius', type=int, default=500, help='Feature extraction radius in meters')
    parser.add_argument('--output', type=str, help='Output path for saved model')
    parser.add_argument('--list', action='store_true', help='List all trained models')
    parser.add_argument('--interactive', action='store_true', help='Run interactive menu')
    
    args = parser.parse_args()
    
    if args.list:
        list_trained_models()
        return
    
    if args.interactive or args.model is None:
        interactive_menu()
        return
    
    # Validate model name
    available_models = get_available_models()
    if args.model not in available_models:
        print(f"❌ Unknown model: {args.model}")
        print(f"Available models: {', '.join(available_models)}")
        return
    
    # Train single model
    result = train_single_model(args.model, args.radius, args.output)
    
    if result:
        print(f"\n🎉 Training completed successfully!")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main()