"""
Enhanced Bike Flow Prediction System
Integrates OSM feature extraction, multi-path routing, and robust ML evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from datetime import datetime
import warnings
import time
import json

# Import our custom modules
from osm_feature_extractor import OSMFeatureExtractor, FeatureInfluenceAnalyzer
from multi_path_router import ImprovedMultiPathRouter as MultiPathRouter
from ml_evaluation_system import MLEvaluationSystem
from optimized_map import create_optimized_map
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Enhanced Bike Flow Prediction System",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: rgba(240, 242, 246, 0.8);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .feature-card {
        background-color: rgba(225, 245, 254, 0.9);
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.3rem 0;
    }
    .evaluation-card {
        background-color: rgba(232, 245, 233, 0.9);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #4caf50; font-weight: bold; }
    .confidence-medium { color: #ff9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_bike_data():
    """Load and preprocess bike trip data"""
    try:
        trips_df = pd.read_csv('data/trips_8days_flat.csv')
        
        # Parse timestamps
        trips_df['start_datetime'] = pd.to_datetime(trips_df['start_time'], format='%Y%m%d_%H%M%S')
        trips_df['end_datetime'] = pd.to_datetime(trips_df['end_time'], format='%Y%m%d_%H%M%S')
        trips_df['hour'] = trips_df['start_datetime'].dt.hour
        trips_df['day_of_week'] = trips_df['start_datetime'].dt.dayofweek
        trips_df['is_weekend'] = trips_df['day_of_week'].isin([5, 6])
        
        # Extract coordinates
        trips_df['start_lat'] = trips_df['start_coords'].apply(lambda x: ast.literal_eval(x)[0])
        trips_df['start_lon'] = trips_df['start_coords'].apply(lambda x: ast.literal_eval(x)[1])
        trips_df['end_lat'] = trips_df['end_coords'].apply(lambda x: ast.literal_eval(x)[0])
        trips_df['end_lon'] = trips_df['end_coords'].apply(lambda x: ast.literal_eval(x)[1])
        
        return trips_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

class EnhancedBikeFlowPredictor:
    """Enhanced predictor with OSM features and multi-path routing"""
    
    def __init__(self, trips_df):
        self.trips_df = trips_df
        self.station_coords = {}
        self.station_features = {}
        self.osm_features = {}
        self.model = None
        self.scaler = None
        self.evaluation_results = None
        
        # Initialize components
        self.osm_extractor = OSMFeatureExtractor()
        self.multi_router = MultiPathRouter()
        self.ml_evaluator = MLEvaluationSystem()
        self.feature_analyzer = FeatureInfluenceAnalyzer()
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the enhanced prediction system"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Build station network
        status_text.text("Building station network...")
        self._build_station_network()
        progress_bar.progress(20)
        
        # Extract OSM features (sample for demo)
        status_text.text("Extracting OSM features...")
        self._extract_osm_features_sample()
        progress_bar.progress(40)
        
        # Engineer enhanced features
        status_text.text("Engineering enhanced features...")
        self._engineer_enhanced_features()
        progress_bar.progress(60)
        
        # Train enhanced model
        status_text.text("Training enhanced ML model...")
        self._train_enhanced_model()
        progress_bar.progress(80)
        
        # Evaluate model
        status_text.text("Evaluating model performance...")
        self._evaluate_model()
        progress_bar.progress(100)
        
        status_text.text("✅ Enhanced system ready!")
        time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
    
    def _build_station_network(self):
        """Build station coordinate mapping"""
        start_stations = self.trips_df.groupby('start_station_id').agg({
            'start_lat': 'first',
            'start_lon': 'first'
        }).reset_index()
        
        for _, row in start_stations.iterrows():
            station_id = row['start_station_id']
            self.station_coords[station_id] = (row['start_lat'], row['start_lon'])
    
    def _extract_osm_features_sample(self):
        """Extract OSM features for a sample of stations (for demo)"""
        # Sample a few stations for demonstration
        sample_stations = list(self.station_coords.keys())[:5]  # First 5 stations
        
        for station_id in sample_stations:
            lat, lon = self.station_coords[station_id]
            
            try:
                # Extract features around this station
                features = self.osm_extractor.extract_features_around_station(
                    lat, lon, radius_m=500
                )
                
                # Compute metrics
                metrics = self.osm_extractor.compute_feature_metrics(features)
                self.osm_features[station_id] = metrics
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.warning(f"Failed to extract OSM features for station {station_id}: {e}")
                self.osm_features[station_id] = {}
    
    def _engineer_enhanced_features(self):
        """Engineer enhanced features including OSM data"""
        for station_id in self.station_coords.keys():
            features = {}
            
            # Basic trip features
            station_trips = self.trips_df[
                (self.trips_df['start_station_id'] == station_id) | 
                (self.trips_df['end_station_id'] == station_id)
            ]
            
            features['total_trips'] = len(station_trips)
            features['avg_temperature'] = station_trips['temperature'].mean() if len(station_trips) > 0 else 20.0
            features['lat'] = self.station_coords[station_id][0]
            features['lon'] = self.station_coords[station_id][1]
            
            # Temporal features
            for hour in range(24):
                outflow = len(station_trips[
                    (station_trips['start_station_id'] == station_id) & 
                    (station_trips['hour'] == hour)
                ])
                features[f'outflow_hour_{hour}'] = outflow
            
            for dow in range(7):
                dow_outflow = len(station_trips[
                    (station_trips['start_station_id'] == station_id) & 
                    (station_trips['day_of_week'] == dow)
                ])
                features[f'outflow_dow_{dow}'] = dow_outflow
            
            # OSM features (if available)
            osm_data = self.osm_features.get(station_id, {})
            for osm_feature, value in osm_data.items():
                features[f'osm_{osm_feature}'] = value
            
            self.station_features[station_id] = features
    
    def _train_enhanced_model(self):
        """Train enhanced model with all features"""
        training_data = []
        
        # Sample training data
        sample_hours = [6, 8, 12, 17, 20]
        sample_days = [0, 1, 4, 5, 6]
        
        for hour in sample_hours:
            for dow in sample_days:
                hour_trips = self.trips_df[
                    (self.trips_df['hour'] == hour) & 
                    (self.trips_df['day_of_week'] == dow)
                ]
                
                flows = hour_trips.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='flow_count')
                
                # Sample for performance
                if len(flows) > 200:
                    flows = flows.sample(n=200, random_state=42)
                
                for _, row in flows.iterrows():
                    start_station = row['start_station_id']
                    end_station = row['end_station_id']
                    flow_count = row['flow_count']
                    
                    if start_station in self.station_features and end_station in self.station_features:
                        feature_vector = self._create_enhanced_feature_vector(
                            start_station, end_station, hour, dow
                        )
                        training_data.append(feature_vector + [flow_count])
        
        if training_data:
            training_data = np.array(training_data)
            X = training_data[:, :-1]
            y = training_data[:, -1]
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X_scaled, y)
    
    def _create_enhanced_feature_vector(self, start_station, end_station, hour, day_of_week):
        """Create enhanced feature vector with OSM features"""
        start_features = self.station_features[start_station]
        end_features = self.station_features[end_station]
        
        # Basic features
        feature_vector = [
            hour,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            day_of_week,
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            1 if day_of_week in [5, 6] else 0,
            start_features['lat'],
            start_features['lon'],
            start_features['total_trips'],
            start_features[f'outflow_hour_{hour}'],
            start_features.get(f'outflow_dow_{day_of_week}', 0),
            end_features['lat'],
            end_features['lon'],
            end_features['total_trips'],
            np.sqrt((start_features['lat'] - end_features['lat'])**2 + 
                   (start_features['lon'] - end_features['lon'])**2),
            (start_features['avg_temperature'] + end_features['avg_temperature']) / 2
        ]
        
        # Add OSM features
        osm_feature_names = [
            'hotels_count', 'restaurants_count', 'banks_count', 'shops_count',
            'schools_count', 'parks_count', 'offices_count', 'residential_count'
        ]
        
        for osm_feature in osm_feature_names:
            start_value = start_features.get(f'osm_{osm_feature}', 0)
            end_value = end_features.get(f'osm_{osm_feature}', 0)
            feature_vector.extend([start_value, end_value])
        
        return feature_vector
    
    def _evaluate_model(self):
        """Evaluate the enhanced model"""
        if self.model is None:
            return
        
        # Prepare evaluation data
        training_data = []
        for hour in [8, 12, 17]:
            for dow in [1, 5]:
                hour_trips = self.trips_df[
                    (self.trips_df['hour'] == hour) & 
                    (self.trips_df['day_of_week'] == dow)
                ]
                
                flows = hour_trips.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='flow_count')
                
                if len(flows) > 100:
                    flows = flows.sample(n=100, random_state=42)
                
                for _, row in flows.iterrows():
                    start_station = row['start_station_id']
                    end_station = row['end_station_id']
                    flow_count = row['flow_count']
                    
                    if start_station in self.station_features and end_station in self.station_features:
                        feature_vector = self._create_enhanced_feature_vector(
                            start_station, end_station, hour, dow
                        )
                        training_data.append(feature_vector + [flow_count])
        
        if training_data:
            training_data = np.array(training_data)
            X = training_data[:, :-1]
            y = training_data[:, -1]
            
            X_scaled = self.scaler.transform(X)
            
            # Get feature names
            feature_names = self._get_feature_names()
            
            # Comprehensive evaluation
            self.evaluation_results = self.ml_evaluator.comprehensive_evaluation(
                self.model, X_scaled, y, feature_names
            )
            
            # Feature influence analysis
            station_features_df = pd.DataFrame.from_dict(self.station_features, orient='index')
            station_features_df['station_id'] = station_features_df.index
            
            # Create flow data for analysis
            flow_data = []
            for _, row in flows.iterrows():
                flow_data.append({
                    'destination_station': row['end_station_id'],
                    'flow_volume': row['flow_count']
                })
            flow_df = pd.DataFrame(flow_data)
            
            if len(flow_df) > 0:
                self.feature_influence = self.feature_analyzer.analyze_feature_influence(
                    station_features_df, flow_df
                )
    
    def _get_feature_names(self):
        """Get feature names for evaluation"""
        base_features = [
            'hour', 'hour_sin', 'hour_cos', 'day_of_week', 'dow_sin', 'dow_cos', 'is_weekend',
            'start_lat', 'start_lon', 'start_total_trips', 'start_outflow_hour', 'start_outflow_dow',
            'end_lat', 'end_lon', 'end_total_trips', 'distance', 'avg_temperature'
        ]
        
        osm_features = []
        osm_feature_names = [
            'hotels_count', 'restaurants_count', 'banks_count', 'shops_count',
            'schools_count', 'parks_count', 'offices_count', 'residential_count'
        ]
        
        for osm_feature in osm_feature_names:
            osm_features.extend([f'start_osm_{osm_feature}', f'end_osm_{osm_feature}'])
        
        return base_features + osm_features
    
    def predict_with_multiple_paths(self, station_id, hour, day_of_week=1, top_k=5):
        """Predict destinations with multiple path options"""
        if station_id not in self.station_features or not self.model or not self.scaler:
            st.warning(f"Missing data for station {station_id}: model={self.model is not None}, scaler={self.scaler is not None}")
            return []
        
        predictions = []
        
        # Get basic predictions
        available_stations = [s for s in self.station_coords.keys() if s != station_id and s in self.station_features]
        
        # Limit for demo but ensure we have some stations
        demo_stations = available_stations[:min(20, len(available_stations))]
        
        for dest_station in demo_stations:
            try:
                feature_vector = self._create_enhanced_feature_vector(
                    station_id, dest_station, hour, day_of_week
                )
                
                if len(feature_vector) == 0:
                    continue
                    
                X_pred = self.scaler.transform([feature_vector])
                predicted_flow = self.model.predict(X_pred)[0]
                
                # Simple confidence calculation
                confidence = min(0.9, max(0.3, abs(predicted_flow) / 10.0))
                
                # Get multiple paths
                start_lat, start_lon = self.station_coords[station_id]
                end_lat, end_lon = self.station_coords[dest_station]
                
                paths = self.multi_router.get_multiple_paths(
                    start_lat, start_lon, end_lat, end_lon, max_paths=3
                )
                
                predictions.append({
                    'destination': dest_station,
                    'predicted_flow': max(1.0, abs(predicted_flow)),
                    'confidence': confidence,
                    'paths': paths
                })
                
            except Exception as e:
                st.warning(f"Prediction failed for {dest_station}: {e}")
                continue
        
        # Sort by predicted flow
        predictions.sort(key=lambda x: x['predicted_flow'], reverse=True)
        return predictions[:top_k]


def main():
    """Main enhanced Streamlit app"""
    
    st.markdown('<h1 class="main-header">🚴‍♂️ Enhanced Bike Flow Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Enhanced Controls")
        
        # Load data
        with st.spinner("Loading bike data..."):
            trips_df = load_bike_data()
            if trips_df is None:
                st.error("Failed to load bike data.")
                return
        
        # Initialize enhanced predictor
        @st.cache_resource
        def get_enhanced_predictor():
            return EnhancedBikeFlowPredictor(trips_df)
        
        predictor = get_enhanced_predictor()
        
        # Controls
        selected_hour = st.slider("🕐 Hour", 0, 23, 17)
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        selected_day_name = st.selectbox("📅 Day of Week", day_names, index=1)
        selected_day_of_week = day_names.index(selected_day_name)
        
        # Station selection
        if 'selected_station' not in st.session_state:
            st.session_state.selected_station = None
        
        available_stations = sorted(list(predictor.station_coords.keys()))
        selected_station = st.selectbox(
            "🚉 Select Station",
            [None] + available_stations,
            index=0
        )
        
        if selected_station != st.session_state.selected_station:
            st.session_state.selected_station = selected_station
        
        # Model evaluation display
        st.markdown("---")
        st.subheader("🎯 Model Performance")
        
        if predictor.evaluation_results:
            summary = predictor.evaluation_results.get('summary', {})
            score = summary.get('overall_score', 0)
            grade = summary.get('confidence_grade', 'Unknown')
            
            # Color code confidence
            if grade == 'High':
                grade_class = 'confidence-high'
            elif grade == 'Medium':
                grade_class = 'confidence-medium'
            else:
                grade_class = 'confidence-low'
            
            st.markdown(f"""
            <div class="evaluation-card">
                <p><strong>Overall Score:</strong> {score:.1f}/100</p>
                <p><strong>Confidence:</strong> <span class="{grade_class}">{grade}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cross-validation metrics
            cv_results = predictor.evaluation_results.get('cross_validation', {})
            if cv_results:
                r2_mean = cv_results.get('r2', {}).get('mean', 0)
                rmse_mean = cv_results.get('rmse', {}).get('mean', 0)
                
                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>R² Score:</strong> {r2_mean:.3f}</p>
                    <p><strong>RMSE:</strong> {rmse_mean:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Feature influence
        if hasattr(predictor, 'feature_influence') and predictor.feature_influence:
            st.subheader("📊 Top Influential Features")
            top_features = list(predictor.feature_influence.items())[:5]
            
            for feature, importance in top_features:
                # Clean feature name
                clean_name = feature.replace('osm_', '').replace('_', ' ').title()
                st.markdown(f"""
                <div class="feature-card">
                    <p><strong>{clean_name}</strong></p>
                    <p>Importance: {importance:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if selected_station:
            st.markdown(f'<p style="color: #1f77b4; font-weight: bold;">🎯 Station {selected_station} - Enhanced Analysis</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="translucent-text">💡 Select a station to see enhanced predictions with multiple paths</p>', unsafe_allow_html=True)
        
        # Create and display optimized map
        map_obj = create_optimized_map(predictor, selected_hour, selected_day_of_week, selected_station)
        map_data = st_folium(map_obj, width=None, height=700)
    
    with col2:
        st.subheader("🔮 Enhanced Predictions")
        
        if selected_station:
            # Station info with OSM features
            osm_features = predictor.osm_features.get(selected_station, {})
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>🚉 Station {selected_station}</h4>
                <p><strong>Time:</strong> {selected_hour:02d}:00 on {day_names[selected_day_of_week]}</p>
                <p><strong>Historical Trips:</strong> {predictor.station_features[selected_station]['total_trips']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # OSM features summary
            if osm_features:
                st.markdown("**🗺️ Nearby Features:**")
                feature_summary = []
                for feature, count in osm_features.items():
                    if 'count' in feature and count > 0:
                        feature_name = feature.replace('_count', '').replace('_', ' ').title()
                        feature_summary.append(f"{feature_name}: {count}")
                
                if feature_summary:
                    for summary in feature_summary[:5]:
                        st.markdown(f"- {summary}")
                else:
                    st.markdown("- No significant features found")
            
            # Enhanced predictions
            with st.spinner("Calculating enhanced predictions..."):
                predictions = predictor.predict_with_multiple_paths(
                    selected_station, selected_hour, day_of_week=selected_day_of_week, top_k=5
                )
            
            if predictions:
                st.markdown("**🎯 Top Destinations with Multiple Paths**")
                
                for i, pred in enumerate(predictions):
                    dest = pred['destination']
                    flow = pred['predicted_flow']
                    confidence = pred['confidence']
                    paths = pred['paths']
                    
                    # Confidence color
                    if confidence > 0.7:
                        conf_class = 'confidence-high'
                    elif confidence > 0.4:
                        conf_class = 'confidence-medium'
                    else:
                        conf_class = 'confidence-low'
                    
                    st.markdown(f"""
                    <div class="feature-card">
                        <h5>#{i+1} Station {dest}</h5>
                        <p><strong>Flow:</strong> {flow:.1f} trips</p>
                        <p><strong>Confidence:</strong> <span class="{conf_class}">{confidence:.0%}</span></p>
                        <p><strong>Paths Available:</strong> {len(paths)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Path details
                    if paths:
                        for j, path in enumerate(paths[:2]):  # Show top 2 paths
                            st.markdown(f"""
                            <div style="margin-left: 1rem; font-size: 0.9rem;">
                                <p><strong>{path.path_type.title()}:</strong> {path.distance_m/1000:.2f}km, {path.duration_s/60:.1f}min</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No enhanced predictions available")
        else:
            st.markdown("**📈 System Overview**")
            
            total_stations = len(predictor.station_coords)
            total_trips = len(trips_df)
            osm_stations = len(predictor.osm_features)
            
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Total Stations:</strong> {total_stations}</p>
                <p><strong>Total Trips:</strong> {total_trips:,}</p>
                <p><strong>OSM Enhanced:</strong> {osm_stations} stations</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Expandable sections
    with st.expander("📋 Enhanced Features & Evaluation Details", expanded=False):
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("""
            **🗺️ OpenStreetMap Features:**
            - Hotels, restaurants, cafes
            - Banks, shops, supermarkets
            - Schools, universities, libraries
            - Parks, sports centers
            - Offices, residential areas
            - Transportation hubs
            
            **🛣️ Multi-Path Routing:**
            - OSRM free routing service
            - GraphHopper free tier
            - Shortest distance paths
            - Fastest time routes
            - Alternative path options
            - Waypoint-based routing
            - Grid-based fallback paths
            """)
        
        with col_right:
            st.markdown("""
            **🎯 ML Evaluation Metrics:**
            - Cross-validation (5-fold)
            - Hold-out testing
            - Time series validation
            - Feature importance analysis
            - Residual analysis
            - Confidence intervals
            - Bootstrap uncertainty
            - Model stability testing
            
            **📊 Confidence Estimation:**
            - Prediction intervals
            - Uncertainty quantification
            - Calibration analysis
            """)
    
    # Model evaluation report
    if predictor.evaluation_results:
        with st.expander("📊 Detailed Model Evaluation Report", expanded=False):
            report = predictor.ml_evaluator.generate_evaluation_report(predictor.evaluation_results)
            st.markdown(report)

if __name__ == "__main__":
    main()