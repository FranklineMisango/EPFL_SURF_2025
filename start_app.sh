#!/bin/bash

echo "🚴‍♂️ Bike Flow Prediction System - All Stations"
echo "=============================================="

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
if [ ! -f ".requirements_installed" ]; then
    echo "📦 Installing dependencies..."
    
    # Fix the requests warning first
    pip install --upgrade requests urllib3 chardet charset-normalizer
    
    # Install other requirements
    pip install -r requirements.txt
    
    # Mark requirements as installed
    touch .requirements_installed
    echo "✅ Dependencies installed and cached"
else
    echo "✅ Dependencies already cached"
fi

echo ""
echo "🚀 Starting Bike Flow Prediction System..."
echo "📍 All stations displayed (may take 15-30 seconds to load)"
echo ""
echo "📍 The app will open at: http://localhost:8501"
echo "🔄 First load may take 15-30 seconds (then cached)"
echo "🎯 Click any station to see predictions!"
echo ""

# Start the app
streamlit run app.py --server.port 8501 --server.address localhost
