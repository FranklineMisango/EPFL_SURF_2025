#!/bin/bash
# fix_time_features.sh - Script to fix the time features mismatch in GNN model

# Create a backup of the original file
echo "Creating backup of run_gnn_baselines.py..."
cp /home/ubuntu/code/EPFL_SURF_2025/helpers/run_gnn_baselines.py /home/ubuntu/code/EPFL_SURF_2025/helpers/run_gnn_baselines.py.bak

# Apply the fix to the evaluate_gnn_model function
echo "Applying fix to the evaluate_gnn_model function..."
sed -i 's/source_indices,\n            target_indices\n        )/source_indices,\n            target_indices,\n            torch.tensor([[sample["time_vec"] for sample in test_samples]], dtype=torch.float32).to(gnn_predictor.device).squeeze(0)\n        )/' /home/ubuntu/code/EPFL_SURF_2025/helpers/run_gnn_baselines.py

# Check for syntax issues
echo "Checking for syntax issues..."
python -c "import sys; sys.path.append('/home/ubuntu/code/EPFL_SURF_2025/helpers'); import run_gnn_baselines"
if [ $? -eq 0 ]; then
  echo "Fix applied successfully!"
else
  echo "Syntax error detected. Restoring backup..."
  cp /home/ubuntu/code/EPFL_SURF_2025/helpers/run_gnn_baselines.py.bak /home/ubuntu/code/EPFL_SURF_2025/helpers/run_gnn_baselines.py
  
  # Try a simpler approach
  echo "Attempting alternate fix..."
  sed -i 's/            pred_flows = gnn_predictor.model(\n                graph_data.x,\n                graph_data.edge_index,\n                graph_data.edge_attr,\n                source_indices,\n                target_indices\n            )/            # Extract time features\n            time_vecs = [sample.get("time_vec", [0.5, 0.5]) for sample in test_samples]\n            time_feats = torch.tensor(time_vecs, dtype=torch.float32).to(gnn_predictor.device)\n            \n            # Get predictions with time features\n            pred_flows = gnn_predictor.model(\n                graph_data.x,\n                graph_data.edge_index,\n                graph_data.edge_attr,\n                source_indices,\n                target_indices,\n                time_feats\n            )/' /home/ubuntu/code/EPFL_SURF_2025/helpers/run_gnn_baselines.py
  
  # Check again
  python -c "import sys; sys.path.append('/home/ubuntu/code/EPFL_SURF_2025/helpers'); import run_gnn_baselines"
  if [ $? -eq 0 ]; then
    echo "Alternate fix applied successfully!"
  else
    echo "Alternate fix failed. Manual editing required."
    cp /home/ubuntu/code/EPFL_SURF_2025/helpers/run_gnn_baselines.py.bak /home/ubuntu/code/EPFL_SURF_2025/helpers/run_gnn_baselines.py
  fi
fi

echo "Done!"
