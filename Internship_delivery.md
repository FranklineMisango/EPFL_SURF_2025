# Enhanced Bike Flow Prediction System 🚴‍♂️

A comprehensive bike flow prediction system that integrates **OpenStreetMap feature extraction**, **multi-path routing**, and **robust ML evaluation** to predict bike destination probabilities with confidence metrics.

## 🌟 Key Features

### 1. 🗺️ OpenStreetMap Feature Extraction
- **40+ Feature Types**: Hotels, restaurants, banks, shops, schools, parks, offices, residential areas, transportation hubs, and more
- **Spatial Analysis**: Features within customizable radius (100m, 250m, 500m bands)
- **Influence Metrics**: Density calculations, distance statistics, and accessibility scores
- **Smart Caching**: Efficient caching system to avoid redundant API calls

### 2. 🛣️ Multi-Path Routing System
- **Multiple Route Types**: Shortest, fastest, safest, scenic, and alternative paths
- **Real Cycling Infrastructure**: Uses actual bike lanes, paths, and cycling-friendly routes
- **Multiple Routing Services**: OpenRouteService, OSRM with intelligent fallbacks
- **Path Diversity Analysis**: Quantifies route variety and overlap metrics

### 3. 🎯 Robust ML Evaluation
- **Comprehensive Metrics**: RMSE, MAE, R², MAPE, correlation analysis
- **Cross-Validation**: K-fold, time series, and hold-out validation
- **Confidence Estimation**: Bootstrap confidence intervals and uncertainty quantification
- **Model Stability**: Performance consistency across different data splits
- **Feature Importance**: Random Forest and permutation importance analysis

### 4. 📊 Feature Influence Analysis
- **Statistical Analysis**: Correlation and significance testing
- **ML-Based Importance**: Random Forest feature importance scores
- **Combined Scoring**: Weighted combination of multiple importance metrics
- **Interpretable Reports**: Detailed analysis of which features drive predictions

## 🏗️ System Architecture

```
Enhanced Bike Flow Prediction System
├── OSM Feature Extractor
│   ├── 40+ feature types (hotels, restaurants, etc.)
│   ├── Spatial metrics (density, distance, accessibility)
│   └── Intelligent caching system
├── Multi-Path Router
│   ├── Multiple routing profiles (safe, fast, scenic)
│   ├── Real cycling infrastructure
│   └── Path diversity analysis
├── ML Evaluation System
│   ├── Comprehensive validation (CV, holdout, time series)
│   ├── Confidence estimation (bootstrap, intervals)
│   └── Model stability analysis
└── Enhanced Predictor
    ├── Temporal features (hour, day-of-week, weekend)
    ├── Spatial features (coordinates, distances)
    ├── OSM features (nearby amenities, land use)
    └── Historical patterns (flow volumes, trends)
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_enhanced.txt

# Optional: Set OpenRouteService API key for better routing
export OPENROUTESERVICE_API_KEY="your_api_key_here"
```

### Run the Enhanced System

```bash
# Run the enhanced Streamlit app
streamlit run enhanced_bike_flow_app.py

# Or run the demo to see all features
python demo_enhanced_system.py
```

### Basic Usage

```python
from enhanced_bike_flow_app import EnhancedBikeFlowPredictor
import pandas as pd

# Load your bike trip data
trips_df = pd.read_csv('data/trips_8days_flat.csv')

# Initialize enhanced predictor
predictor = EnhancedBikeFlowPredictor(trips_df)

# Get predictions with multiple paths
predictions = predictor.predict_with_multiple_paths(
    station_id=123,
    hour=17,
    day_of_week=1,  # Tuesday
    top_k=5
)

# Each prediction includes:
# - destination station
# - predicted flow volume
# - confidence score
# - multiple path options with real routing
```

## 📋 Feature Categories

### 🏪 Commercial & Services
- **Hotels**: Tourist destinations, temporary stays
- **Restaurants**: Meal destinations, social gatherings  
- **Banks**: Financial errands, business trips
- **Shops**: Shopping trips, daily errands
- **Supermarkets**: Regular shopping, groceries
- **Pharmacies**: Health-related trips

### 🚌 Transportation
- **Bus Stops**: Multimodal transport hubs
- **Train Stations**: Major transport connections
- **Parking**: Car-bike transfer points

### 🎓 Education & Culture
- **Schools**: Student commutes, family trips
- **Universities**: Student/staff commutes
- **Libraries**: Study destinations
- **Museums**: Cultural visits, tourism

### 🏥 Healthcare
- **Hospitals**: Medical appointments, emergencies
- **Clinics**: Healthcare visits
- **Dentists**: Medical appointments

### 🌳 Recreation & Sports
- **Parks**: Recreation, exercise, relaxation
- **Sports Centers**: Fitness activities
- **Swimming Pools**: Recreation, fitness
- **Playgrounds**: Family activities

### 🏢 Work & Business
- **Offices**: Work commutes, business meetings
- **Coworking Spaces**: Flexible work locations

### 🏘️ Land Use Zones
- **Residential**: Home origins/destinations
- **Commercial**: Business activities
- **Industrial**: Work commutes, logistics
- **Retail**: Shopping destinations

## 🎯 ML Evaluation Metrics

### Core Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error
- **Explained Variance**: Model explanation power

### Validation Methods
- **K-Fold Cross-Validation**: 5-fold validation with stratification
- **Time Series Validation**: Temporal split validation
- **Hold-out Testing**: 80/20 train/test split
- **Bootstrap Validation**: Uncertainty estimation

### Confidence Metrics
- **Prediction Intervals**: 90% and 95% confidence intervals
- **Uncertainty Quantification**: Bootstrap-based uncertainty
- **Coverage Analysis**: Interval coverage validation
- **Calibration Scores**: Prediction reliability assessment

### Model Stability
- **Performance Consistency**: Across different data splits
- **Feature Stability**: Importance ranking consistency
- **Robustness Testing**: Performance under data variations

## 🛣️ Multi-Path Routing

### Routing Profiles
- **cycling-regular**: Standard cycling route (balanced)
- **cycling-safe**: Safest route with bike infrastructure
- **cycling-fast**: Fastest time route
- **cycling-scenic**: Scenic route through parks
- **cycling-direct**: Most direct distance route

### Path Types
- **Shortest**: Minimum distance path
- **Fastest**: Minimum time path
- **Safest**: Maximum safety (bike lanes, low traffic)
- **Alternative 1**: First alternative route
- **Alternative 2**: Second alternative route
- **Scenic**: Route through parks and green spaces

### Routing Services
1. **OpenRouteService**: Professional routing with cycling profiles
2. **OSRM**: Open Source Routing Machine
3. **Synthetic Generation**: Intelligent fallback with waypoints
4. **Smart Caching**: Efficient route storage and retrieval

## 📊 Usage Examples

### 1. Extract OSM Features

```python
from osm_feature_extractor import OSMFeatureExtractor

extractor = OSMFeatureExtractor()

# Extract features around a location
features = extractor.extract_features_around_station(
    lat=46.5197, 
    lon=6.6323, 
    radius_m=500
)

# Compute aggregated metrics
metrics = extractor.compute_feature_metrics(features)
print(f"Hotels nearby: {metrics.get('hotels_count', 0)}")
print(f"Restaurants nearby: {metrics.get('restaurants_count', 0)}")
```

### 2. Multi-Path Routing

```python
from multi_path_router import MultiPathRouter

router = MultiPathRouter()

# Get multiple paths between two points
paths = router.get_multiple_paths(
    start_lat=46.5197, start_lon=6.6323,
    end_lat=46.5238, end_lon=6.6356,
    max_paths=5
)

for path in paths:
    print(f"{path.path_type}: {path.distance_m/1000:.2f}km, {path.duration_s/60:.1f}min")
```

### 3. ML Evaluation

```python
from ml_evaluation_system import MLEvaluationSystem
from sklearn.ensemble import RandomForestRegressor

evaluator = MLEvaluationSystem()

# Comprehensive model evaluation
results = evaluator.comprehensive_evaluation(
    model=RandomForestRegressor(),
    X=your_features,
    y=your_targets,
    feature_names=feature_names
)

print(f"Overall Score: {results['summary']['overall_score']:.1f}/100")
print(f"Confidence: {results['summary']['confidence_grade']}")
```

### 4. Feature Influence Analysis

```python
from osm_feature_extractor import FeatureInfluenceAnalyzer

analyzer = FeatureInfluenceAnalyzer()

# Analyze which features influence bike flows
influence_scores = analyzer.analyze_feature_influence(
    station_features_df, 
    flow_data_df
)

# Get top influential features
top_features = analyzer.get_top_influential_features(10)
for feature, score in top_features:
    print(f"{feature}: {score:.4f}")
```

## 🎮 Interactive Features

### Enhanced Map Visualization
- **Satellite & Street Views**: Multiple map layers
- **Station Markers**: Color-coded by selection status
- **Multiple Path Lines**: Different colors and styles for each route type
- **OSM Feature Overlays**: Visual representation of nearby amenities
- **Interactive Popups**: Detailed information on click

### Real-time Controls
- **Hour Slider**: 24-hour time selection
- **Day of Week**: Weekday vs weekend patterns
- **Station Selection**: Click-to-select or dropdown
- **Path Type Filtering**: Show/hide different route types

### Performance Dashboard
- **Model Metrics**: Real-time evaluation scores
- **Confidence Grades**: High/Medium/Low confidence indicators
- **Feature Importance**: Top influential features display
- **Prediction Quality**: Uncertainty and interval coverage

## 🔧 Configuration

### Environment Variables
```bash
# Optional: OpenRouteService API key for enhanced routing
export OPENROUTESERVICE_API_KEY="your_api_key"

# Cache directories (optional)
export OSM_CACHE_DIR="cache/osm_features"
export ROUTING_CACHE_DIR="cache/multi_paths"
```

### Customization Options
- **Feature Extraction Radius**: Adjust search radius for OSM features
- **Routing Profiles**: Add custom cycling profiles
- **Evaluation Metrics**: Configure validation methods
- **Caching Strategy**: Adjust cache retention and cleanup

## 📈 Performance Optimization

### Caching Strategy
- **OSM Features**: 24-hour cache retention
- **Routing Data**: Persistent cache with smart invalidation
- **Model Results**: Session-based caching for predictions

### Scalability Features
- **Batch Processing**: Efficient bulk feature extraction
- **Parallel Routing**: Concurrent path generation
- **Smart Sampling**: Intelligent data sampling for large datasets
- **Progressive Loading**: Incremental feature loading

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd EPFL_SURF_2025

# Install development dependencies
pip install -r requirements_enhanced.txt

# Run tests
python demo_enhanced_system.py

# Start development server
streamlit run enhanced_bike_flow_app.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenStreetMap**: For comprehensive geospatial data
- **OpenRouteService**: For professional routing services
- **OSRM**: For open-source routing capabilities
- **Streamlit**: For interactive web application framework
- **Scikit-learn**: For machine learning capabilities

## 📞 Support

For questions, issues, or contributions:
- **Create an Issue**: Use GitHub issues for bug reports
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check this README and code comments

---

**Built with ❤️ for the EPFL SURF 2025 program**