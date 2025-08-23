#!/bin/bash
# Comprehensive training script for Flow Prediction Lab
# Supports all 16 model types with individual training and pickle serialization

echo "Flow Prediction Model Trainer"
echo "================================="

# Check if data files exist
if [ ! -f "data/trips_8days_flat.csv" ]; then
    echo "❌ Error: data/trips_8days_flat.csv not found"
    echo "💡 Please ensure your data files are in the data/ directory"
    exit 1
fi

if [ ! -f "data/unique_stations.csv" ]; then
    echo "❌ Error: data/unique_stations.csv not found"
    echo "💡 Please ensure your data files are in the data/ directory"
    exit 1
fi

# Check for OSM features
if [ ! -d "cache/ml_ready" ] || [ -z "$(ls -A cache/ml_ready 2>/dev/null)" ]; then
    echo "⚠️  Warning: No OSM features found in cache/ml_ready/"
    echo "💡 Consider running: python setup_osm_cache.py download"
    echo "   This will improve model performance with external features"
fi

# Function to show available models
show_models() {
    echo ""
    echo "📋 Available Models:"
    echo "  1. DCRNN                    9. RandomForest"
    echo "  2. GCN                     10. MLP"
    echo "  3. GraphSAGE               11. TabNet"
    echo "  4. GAT                     12. Stacking"
    echo "  5. GIN                     13. Blending"
    echo "  6. Transformer             14. ST-GCN"
    echo "  7. XGBoost                 15. TemporalFusionTransformer"
    echo "  8. LightGBM                16. CatBoost"
    echo ""
}

# Parse command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [MODEL_NAME] [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --interactive    Run interactive menu"
    echo "  --list          List trained models"
    echo "  --all           Train all models (warning: takes a long time)"
    echo "  --help          Show this help"
    show_models
    echo "Examples:"
    echo "  $0 XGBoost              # Train XGBoost model"
    echo "  $0 --interactive        # Interactive menu"
    echo "  $0 --list              # List trained models"
    exit 0
fi

# Handle different modes
if [ "$1" = "--list" ]; then
    echo "📊 Listing trained models..."
    python train_single_model.py --list
    exit 0
elif [ "$1" = "--interactive" ] || [ -z "$1" ]; then
    echo "🎮 Starting interactive mode..."
    python train_single_model.py --interactive
    exit 0
elif [ "$1" = "--all" ]; then
    echo "⚠️  Training ALL models - this will take a very long time!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Cancelled"
        exit 1
    fi
    
    # Train all models
    models=("XGBoost" "LightGBM" "CatBoost" "RandomForest" "MLP" "GCN" "GraphSAGE")
    for model in "${models[@]}"; do
        echo "🔄 Training $model..."
        python train_single_model.py --model "$model"
    done
    
    echo "🎉 All models training completed!"
    python train_single_model.py --list
    exit 0
else
    # Train specific model
    MODEL_NAME="$1"
    RADIUS="${2:-500}"
    
    echo "🔄 Training $MODEL_NAME model with ${RADIUS}m radius..."
    echo "⏱️  This may take a few minutes depending on the model complexity"
    
    python train_single_model.py --model "$MODEL_NAME" --radius "$RADIUS"
    
    # Convert model name to lowercase for file check
    MODEL_FILE=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')_model.pkl
    
    # Check if model was created
    if [ -f "models/$MODEL_FILE" ]; then
        echo "✅ Training complete! Model saved to models/$MODEL_FILE"
        echo "📊 You can now use this model in your applications"
        echo "💡 Run '$0 --list' to see all trained models"
    else
        echo "❌ Training may have failed - check the output above"
        echo "💡 Available models directory contents:"
        ls -la models/ 2>/dev/null || echo "   No models directory found"
        exit 1
    fi
fi