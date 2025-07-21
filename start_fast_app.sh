#!/bin/bash

echo "🚴‍♂️ Fast Bike Flow Prediction System"
echo "====================================="

# Check if data directory exists
if [ ! -d "data" ] || [ ! -f "data/trips_8days_flat.csv" ]; then
    echo "❌ Error: data/trips_8days_flat.csv not found!"
    echo "Please ensure the data file is in the data/ directory."
    exit 1
fi

echo "✅ Data file found"

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f ".fast_requirements_installed" ]; then
    echo "📦 Installing optimized dependencies..."
    pip install -r requirements_fast.txt
    
    # Mark requirements as installed
    touch .fast_requirements_installed
    echo "✅ Dependencies installed and cached"
else
    echo "✅ Dependencies already cached"
fi

echo ""
echo "🚀 Starting Fast Bike Flow Prediction System..."
echo "⚡ Optimized for speed with smart caching"
echo ""
echo "📍 The app will open at: http://localhost:8501"
echo "🔄 Loading may take 10-15 seconds on first run (then cached)"
echo ""

# Start the app
streamlit run fast_app.py --server.port 8501 --server.address localhost
