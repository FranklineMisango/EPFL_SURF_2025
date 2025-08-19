"""
Fixed version of evaluate_gnn_model function for run_gnn_baselines.py
"""
import torch
import numpy as np

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
        
        # Extract time features
        time_vecs = [sample.get("time_vec", [0.5, 0.5]) for sample in test_samples]
        time_feats = torch.tensor(time_vecs, dtype=torch.float32).to(gnn_predictor.device)
        
        # Get predictions with time features
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
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
