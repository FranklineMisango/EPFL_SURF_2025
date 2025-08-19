#!/bin/bash
# run_training.sh - Script to run GNN training with checkpointing for spot instance resilience

# Set default parameters
MODEL_TYPE=${1:-"GCN"}  # Default to GCN if not specified
EPOCHS=${2:-100}
LEARNING_RATE=${3:-0.001}
CHECKPOINT_INTERVAL=${4:-5}  # Save checkpoint every 5 epochs by default

# Create results directory if it doesn't exist
mkdir -p /home/ubuntu/EPFL_SURF_2025/results/checkpoints

# Check if previous checkpoint exists
CHECKPOINT_PATH="/home/ubuntu/EPFL_SURF_2025/results/checkpoints/${MODEL_TYPE}_latest.pt"
RESUME_FLAG=""

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found checkpoint at $CHECKPOINT_PATH, resuming training..."
    RESUME_FLAG="--resume $CHECKPOINT_PATH"
fi

# Run the training script with parameters
echo "Starting training with model: $MODEL_TYPE, epochs: $EPOCHS, learning rate: $LEARNING_RATE"
cd /home/ubuntu/EPFL_SURF_2025

# Set environment variable to use GPU
export CUDA_VISIBLE_DEVICES=0

# Run the GNN training with checkpointing
python run_gnn.py \
    --model_type $MODEL_TYPE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --save_path "/home/ubuntu/EPFL_SURF_2025/results/checkpoints" \
    $RESUME_FLAG

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    # Copy final results to a more permanent location (could be S3, etc.)
    cp /home/ubuntu/EPFL_SURF_2025/results/checkpoints/${MODEL_TYPE}_final.pt /home/ubuntu/EPFL_SURF_2025/results/${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).pt
else
    echo "Training was interrupted. Checkpoint should be saved at $CHECKPOINT_PATH"
fi
