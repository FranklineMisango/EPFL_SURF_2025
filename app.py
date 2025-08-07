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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from osm_feature_extractor import OSMFeatureExtractor, FeatureInfluenceAnalyzer
from multi_path_router import ImprovedMultiPathRouter as MultiPathRouter, PathInfo
from ml_evaluation_system import MLEvaluationSystem
from optimized_map import create_optimized_map
from population_feature_extractor import PopulationFeatureExtractor, EnhancedOSMFeatureExtractor
from multi_stop_journey_predictor import MultiStopJourneyPredictor, POI, JourneySegment, MultiStopJourney
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
    """Enhanced predictor with multi-stop journey prediction and POI analysis"""
    
    def __init__(self, trips_df):
        self.trips_df = trips_df
        self.station_coords = {}
        self.station_features = {}
        self.osm_features = {}
        self.population_features = {}
        self.comprehensive_features = {}
        self.model = None
        self.scaler = None
        self.evaluation_results = None
        
        # Initialize components
        self.osm_extractor = OSMFeatureExtractor()
        self.population_extractor = PopulationFeatureExtractor()
        self.enhanced_extractor = EnhancedOSMFeatureExtractor()
        self.multi_router = MultiPathRouter()
        self.ml_evaluator = MLEvaluationSystem()
        self.feature_analyzer = FeatureInfluenceAnalyzer()
        
        # New multi-stop journey predictor
        self.journey_predictor = None
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the enhanced prediction system"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Build station network
        status_text.text("Building station network...")
        self._build_station_network()
        progress_bar.progress(10)
        
        # Check population data availability
        status_text.text("Checking population data...")
        self._check_population_data()
        progress_bar.progress(20)
        
        # Extract OSM features (sample for demo)
        status_text.text("Extracting OSM features...")
        self._extract_osm_features_sample()
        progress_bar.progress(35)
        
        # Extract population features (sample for demo)
        status_text.text("Extracting population features...")
        self._extract_population_features_sample()
        progress_bar.progress(50)
        
        # Initialize multi-stop journey predictor
        status_text.text("Building POI database and journey predictor...")
        self._initialize_journey_predictor()
        progress_bar.progress(65)
        
        # Engineer enhanced features
        status_text.text("Engineering comprehensive features...")
        self._engineer_enhanced_features()
        progress_bar.progress(75)
        
        # Train enhanced model
        status_text.text("Training enhanced ML model...")
        self._train_enhanced_model()
        progress_bar.progress(85)
        
        # Evaluate model
        status_text.text("Evaluating model performance...")
        self._evaluate_model()
        progress_bar.progress(100)
        
        status_text.text("✅ Enhanced multi-stop journey system ready!")
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
    
    def _check_population_data(self):
        """Check population data availability and display summary"""
        pop_summary = self.population_extractor.get_population_summary()
        
        if pop_summary['data_loaded']:
            st.info(f"📊 Population data loaded: {pop_summary['centroids_count']:,} populated cells, "
                   f"Total population: {pop_summary['total_population']:,.0f}")
        else:
            st.warning("⚠️ Population data not available - continuing with OSM features only")
    
    def _extract_population_features_sample(self):
        """Extract population features for a sample of stations (for demo)"""
        # Sample a few stations for demonstration
        sample_stations = list(self.station_coords.keys())[:8]  # First 8 stations
        
        for station_id in sample_stations:
            lat, lon = self.station_coords[station_id]
            
            try:
                # Extract population features around this station
                pop_features = self.population_extractor.extract_population_features_around_station(
                    lat, lon, radius_m=500
                )
                
                self.population_features[station_id] = pop_features
                
                # Also extract comprehensive features (OSM + Population + Interactions)
                comprehensive = self.enhanced_extractor.extract_comprehensive_features(
                    lat, lon, radius_m=500
                )
                
                self.comprehensive_features[station_id] = comprehensive
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.warning(f"Failed to extract population features for station {station_id}: {e}")
                self.population_features[station_id] = {}
                self.comprehensive_features[station_id] = {}
    
    def _initialize_journey_predictor(self):
        """Initialize the multi-stop journey predictor"""
        try:
            # Initialize journey predictor
            self.journey_predictor = MultiStopJourneyPredictor(
                self.osm_extractor,
                self.population_extractor, 
                self.multi_router
            )
            
            # Build POI database (sample for demo)
            sample_stations = {k: v for k, v in list(self.station_coords.items())[:10]}  # First 10 stations
            poi_database = self.journey_predictor.build_poi_database(sample_stations, radius_m=1000)
            
            # Learn journey patterns
            self.journey_predictor.learn_journey_patterns(self.trips_df, sample_stations)
            
            st.info(f"🗺️ Built POI database with {len(poi_database)} Points of Interest")
            
        except Exception as e:
            st.warning(f"Failed to initialize journey predictor: {e}")
            self.journey_predictor = None
    
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
        """Engineer enhanced features including OSM data and population data"""
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
            
            # Population features (if available)
            pop_data = self.population_features.get(station_id, {})
            for pop_feature, value in pop_data.items():
                features[f'pop_{pop_feature}'] = value
            
            # Comprehensive features (OSM + Population + Interactions)
            comp_data = self.comprehensive_features.get(station_id, {})
            for comp_feature, value in comp_data.items():
                # Avoid duplicating OSM and pop features we already added
                if not comp_feature.startswith(('osm_', 'pop_')):
                    features[f'comp_{comp_feature}'] = value
                elif comp_feature not in features:  # Add if not already present
                    features[comp_feature] = value
            
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
        """Create enhanced feature vector with OSM and population features"""
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
        
        # Add population features
        pop_feature_names = [
            'total_population', 'population_density_km2', 'populated_cells_count',
            'population_within_500m', 'density_within_500m', 'population_gini'
        ]
        
        for pop_feature in pop_feature_names:
            start_value = start_features.get(f'pop_{pop_feature}', 0)
            end_value = end_features.get(f'pop_{pop_feature}', 0)
            feature_vector.extend([start_value, end_value])
        
        # Add interaction features
        interaction_feature_names = [
            'restaurants_per_1000_people', 'shops_per_1000_people', 
            'amenity_diversity_score', 'transit_per_1000_people'
        ]
        
        for interaction_feature in interaction_feature_names:
            start_value = start_features.get(f'comp_{interaction_feature}', 0)
            end_value = end_features.get(f'comp_{interaction_feature}', 0)
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
        
        # Population features
        pop_features = []
        pop_feature_names = [
            'total_population', 'population_density_km2', 'populated_cells_count',
            'population_within_500m', 'density_within_500m', 'population_gini'
        ]
        
        for pop_feature in pop_feature_names:
            pop_features.extend([f'start_pop_{pop_feature}', f'end_pop_{pop_feature}'])
        
        # Interaction features
        interaction_features = []
        interaction_feature_names = [
            'restaurants_per_1000_people', 'shops_per_1000_people', 
            'amenity_diversity_score', 'transit_per_1000_people'
        ]
        
        for interaction_feature in interaction_feature_names:
            interaction_features.extend([f'start_comp_{interaction_feature}', f'end_comp_{interaction_feature}'])
        
        return base_features + osm_features + pop_features + interaction_features
    
    def predict_multi_stop_journeys(self, station_id, hour, day_of_week=1, top_destinations=3):
        """Predict multi-stop journeys with POI visits"""
        if not self.journey_predictor or station_id not in self.station_coords:
            st.warning(f"Journey predictor not available or station {station_id} not found")
            return []
        
        # Get available destination stations
        available_destinations = [s for s in self.station_coords.keys() 
                                if s != station_id and s in self.journey_predictor.station_coverage]
        
        if not available_destinations:
            return []
        
        # Limit destinations for demo
        demo_destinations = available_destinations[:min(top_destinations, len(available_destinations))]
        
        all_journeys = []
        
        for dest_station in demo_destinations:
            try:
                # Predict multi-stop journeys to this destination
                journeys = self.journey_predictor.predict_multi_stop_journey(
                    station_id, dest_station, hour, day_of_week,
                    max_stops=2,  # Limit to 2 stops for demo
                    max_detour_factor=2.5
                )
                
                all_journeys.extend(journeys)
                
            except Exception as e:
                logger.warning(f"Failed to predict journey to {dest_station}: {e}")
                continue
        
        # Sort by visit probability and return top journeys
        all_journeys.sort(key=lambda j: j.visit_probability, reverse=True)
        return all_journeys[:5]  # Top 5 journey options
    
    def predict_with_multiple_paths(self, station_id, hour, day_of_week=1, top_k=5):
        """Predict flows with multiple path options for optimized map visualization"""
        if not self.model or not self.scaler or station_id not in self.station_coords:
            return []
        
        predictions = []
        
        # Get potential destination stations
        available_destinations = [s for s in self.station_coords.keys() if s != station_id]
        
        # Limit for performance
        sample_destinations = available_destinations[:min(top_k, len(available_destinations))]
        
        for dest_station in sample_destinations:
            if dest_station in self.station_features:
                try:
                    # Create feature vector for prediction
                    feature_vector = self._create_enhanced_feature_vector(
                        station_id, dest_station, hour, day_of_week
                    )
                    
                    # Get flow prediction
                    X_scaled = self.scaler.transform([feature_vector])
                    predicted_flow = max(0, self.model.predict(X_scaled)[0])
                    
                    # Calculate confidence based on model uncertainty
                    confidence = min(1.0, predicted_flow / 10.0)  # Simple confidence metric
                    
                    # Get multiple paths using multi-path router
                    start_coords = self.station_coords[station_id]
                    end_coords = self.station_coords[dest_station]
                    
                    paths = []
                    try:
                        # Get multiple routing options
                        routing_result = self.multi_router.get_multiple_paths(
                            start_coords[0], start_coords[1],
                            end_coords[0], end_coords[1],
                            max_paths=3
                        )
                        
                        # The method returns a list of PathInfo objects directly
                        if routing_result:
                            paths = routing_result
                    except Exception as e:
                        logger.warning(f"Failed to get routes for {station_id} -> {dest_station}: {e}")
                        # Create a simple fallback PathInfo object
                        fallback_path = PathInfo(
                            path_id=f"fallback_{station_id}_{dest_station}",
                            coordinates=[start_coords, end_coords],
                            distance_m=1000.0,
                            duration_s=600.0,
                            path_type='direct',
                            routing_profile='cycling-regular'
                        )
                        paths = [fallback_path]
                    
                    predictions.append({
                        'destination': dest_station,
                        'predicted_flow': predicted_flow,
                        'confidence': confidence,
                        'paths': paths
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to predict for {station_id} -> {dest_station}: {e}")
                    continue
        
        # Sort by predicted flow
        predictions.sort(key=lambda x: x['predicted_flow'], reverse=True)
        return predictions


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
        def get_enhanced_predictor(_version="v1.3"):  # Updated version to force cache refresh after routing fix
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
            # Station info with OSM and population features
            osm_features = predictor.osm_features.get(selected_station, {})
            pop_features = predictor.population_features.get(selected_station, {})
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>🚉 Station {selected_station}</h4>
                <p><strong>Time:</strong> {selected_hour:02d}:00 on {day_names[selected_day_of_week]}</p>
                <p><strong>Historical Trips:</strong> {predictor.station_features[selected_station]['total_trips']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Population features summary
            if pop_features:
                st.markdown("**👥 Population Context:**")
                
                total_pop = pop_features.get('total_population', 0)
                pop_density = pop_features.get('population_density_km2', 0)
                pop_cells = pop_features.get('populated_cells_count', 0)
                
                st.markdown(f"""
                <div class="feature-card">
                    <p><strong>Population (500m):</strong> {total_pop:,.0f} people</p>
                    <p><strong>Density:</strong> {pop_density:,.0f} people/km²</p>
                    <p><strong>Populated Cells:</strong> {pop_cells}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # OSM features summary
            if osm_features:
                st.markdown("**🗺️ Nearby Amenities:**")
                feature_summary = []
                for feature, count in osm_features.items():
                    if 'count' in feature and count > 0:
                        feature_name = feature.replace('_count', '').replace('_', ' ').title()
                        feature_summary.append(f"{feature_name}: {count}")
                
                if feature_summary:
                    for summary in feature_summary[:5]:
                        st.markdown(f"- {summary}")
                else:
                    st.markdown("- No significant amenities found")
            
            # Interaction features (if available)
            comp_features = predictor.comprehensive_features.get(selected_station, {})
            if comp_features and any(key for key in comp_features.keys() if not key.startswith(('osm_', 'pop_'))):
                st.markdown("**🔄 Population-Amenity Insights:**")
                
                restaurants_per_cap = comp_features.get('restaurants_per_1000_people', 0)
                shops_per_cap = comp_features.get('shops_per_1000_people', 0)
                transit_access = comp_features.get('transit_per_1000_people', 0)
                
                if restaurants_per_cap > 0 or shops_per_cap > 0 or transit_access > 0:
                    st.markdown(f"""
                    <div class="feature-card">
                        <p><strong>Restaurants per 1K people:</strong> {restaurants_per_cap:.1f}</p>
                        <p><strong>Shops per 1K people:</strong> {shops_per_cap:.1f}</p>
                        <p><strong>Transit per 1K people:</strong> {transit_access:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Multi-stop journey predictions
            with st.spinner("Calculating multi-stop journey predictions..."):
                journeys = predictor.predict_multi_stop_journeys(
                    selected_station, selected_hour, day_of_week=selected_day_of_week, top_destinations=3
                )
            
            if journeys:
                st.markdown("**🛤️ Multi-Stop Journey Options**")
                
                for i, journey in enumerate(journeys):
                    # Journey header
                    journey_emoji = {
                        'direct': '🎯',
                        'errands': '🛍️', 
                        'lunch': '🍽️',
                        'dining': '🍷',
                        'leisure': '🎨',
                        'work': '💼',
                        'health': '🏥'
                    }.get(journey.journey_type, '🚴‍♂️')
                    
                    # Confidence color
                    if journey.visit_probability > 0.7:
                        conf_class = 'confidence-high'
                    elif journey.visit_probability > 0.4:
                        conf_class = 'confidence-medium'
                    else:
                        conf_class = 'confidence-low'
                    
                    st.markdown(f"""
                    <div class="feature-card">
                        <h5>{journey_emoji} Journey #{i+1} to Station {journey.destination_station}</h5>
                        <p><strong>Type:</strong> {journey.journey_type.title()}</p>
                        <p><strong>Distance:</strong> {journey.total_distance_m/1000:.2f}km</p>
                        <p><strong>Duration:</strong> {journey.total_duration_minutes:.0f} minutes</p>
                        <p><strong>Likelihood:</strong> <span class="{conf_class}">{journey.visit_probability:.0%}</span></p>
                        <p><strong>Segments:</strong> {len(journey.segments)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show POI stops if any
                    poi_stops = [seg.poi for seg in journey.segments if seg.poi is not None]
                    if poi_stops:
                        st.markdown("**📍 POI Stops:**")
                        for poi in poi_stops:
                            poi_emoji = {
                                'restaurants': '🍽️',
                                'cafes': '☕',
                                'shops': '🛍️',
                                'supermarkets': '🛒',
                                'banks': '🏦',
                                'atms': '💳',
                                'schools': '🏫',
                                'parks': '🌳',
                                'hospitals': '🏥',
                                'pharmacies': '💊'
                            }.get(poi.poi_type, '📍')
                            
                            st.markdown(f"""
                            <div style="margin-left: 1rem; font-size: 0.9rem; margin-bottom: 0.5rem;">
                                <p>{poi_emoji} <strong>{poi.name}</strong> ({poi.poi_type.replace('_', ' ').title()})</p>
                                <p style="margin-left: 1.5rem;">Popularity: {poi.popularity_score:.0%} | Visit: ~{predictor.journey_predictor._estimate_poi_visit_duration(poi, selected_hour, selected_day_of_week):.0f}min</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show segment details (collapsible)
                    with st.expander(f"🗺️ Route Details for Journey {i+1}", expanded=False):
                        for j, segment in enumerate(journey.segments):
                            segment_type_emoji = {
                                'direct': '➡️',
                                'to_poi': '📍',
                                'from_poi': '🔄',
                                'to_destination': '🎯'
                            }.get(segment.segment_type, '📍')
                            
                            st.markdown(f"""
                            **{segment_type_emoji} Segment {j+1}:** {segment.segment_type.replace('_', ' ').title()}
                            - Distance: {segment.distance_m/1000:.2f}km
                            - Duration: {segment.duration_minutes:.0f} minutes
                            - From: ({segment.from_lat:.4f}, {segment.from_lon:.4f})
                            - To: ({segment.to_lat:.4f}, {segment.to_lon:.4f})
                            """)
            else:
                st.warning("⚠️ No multi-stop journey predictions available")
        else:
            st.markdown("**📈 System Overview**")
            
            total_stations = len(predictor.station_coords)
            total_trips = len(trips_df)
            osm_stations = len(predictor.osm_features)
            pop_stations = len(predictor.population_features)
            comp_stations = len(predictor.comprehensive_features)
            
            # Get population data summary
            pop_summary = predictor.population_extractor.get_population_summary()
            
            # Get journey predictor summary
            journey_summary = {}
            if predictor.journey_predictor:
                journey_summary = predictor.journey_predictor.get_journey_summary()
            
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>Total Stations:</strong> {total_stations}</p>
                <p><strong>Total Trips:</strong> {total_trips:,}</p>
                <p><strong>OSM Enhanced:</strong> {osm_stations} stations</p>
                <p><strong>Population Enhanced:</strong> {pop_stations} stations</p>
                <p><strong>Comprehensive Features:</strong> {comp_stations} stations</p>
            </div>
            """, unsafe_allow_html=True)
            
            if pop_summary['data_loaded']:
                st.markdown(f"""
                <div class="feature-card">
                    <h5>📊 Population Dataset</h5>
                    <p><strong>Grid Cells:</strong> {pop_summary['centroids_count']:,}</p>
                    <p><strong>Total Population:</strong> {pop_summary['total_population']:,.0f}</p>
                    <p><strong>Avg per Cell:</strong> {pop_summary['average_population_cell']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if journey_summary:
                st.markdown(f"""
                <div class="feature-card">
                    <h5>🗺️ POI & Journey System</h5>
                    <p><strong>Points of Interest:</strong> {journey_summary.get('poi_count', 0):,}</p>
                    <p><strong>Station Coverage:</strong> {journey_summary.get('station_coverage', 0)}</p>
                    <p><strong>POI Network Nodes:</strong> {journey_summary.get('poi_graph_nodes', 0)}</p>
                    <p><strong>POI Connections:</strong> {journey_summary.get('poi_graph_edges', 0)}</p>
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
            
            **� Population Features:**
            - Population density within radius
            - Demographic distribution
            - Population concentration metrics
            - Distance-weighted demographics
            - Population center analysis
            - Grid cell coverage statistics
            
            **🔄 Population-Amenity Interactions:**
            - Amenities per capita ratios
            - Population-weighted accessibility
            - Transit accessibility per population
            - Service density analysis
            
            **�🛣️ Multi-Path Routing:**
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
            
            **�‍♂️ Journey Pattern Learning:**
            - Temporal pattern analysis
            - Spatial route preferences
            - POI visit probability models
            - Journey type classification
            - Duration prediction models
            - Population-context routing
            
            **�📊 Confidence Estimation:**
            - Prediction intervals
            - Uncertainty quantification
            - Calibration analysis
            - Multi-model ensemble
            """)
    
    # Model evaluation report
    if predictor.evaluation_results:
        with st.expander("📊 Detailed Model Evaluation Report", expanded=False):
            report = predictor.ml_evaluator.generate_evaluation_report(predictor.evaluation_results)
            st.markdown(report)
    
    # POI Analysis Report
    if predictor.journey_predictor:
        with st.expander("🗺️ POI Network Analysis", expanded=False):
            journey_summary = predictor.journey_predictor.get_journey_summary()
            
            st.markdown(f"""
            ## POI Database Analysis
            
            **Database Statistics:**
            - Total POIs: {journey_summary.get('poi_count', 0):,}
            - Station Coverage: {journey_summary.get('station_coverage', 0)} stations
            - POI Network Nodes: {journey_summary.get('poi_graph_nodes', 0):,}
            - POI Connections: {journey_summary.get('poi_graph_edges', 0):,}
            
            **Model Training Status:**
            - POI Visit Model: {'✅ Trained' if journey_summary.get('models_trained', {}).get('poi_visit_model', False) else '❌ Not trained'}
            - Journey Type Model: {'✅ Trained' if journey_summary.get('models_trained', {}).get('journey_type_model', False) else '❌ Not trained'}
            - Duration Model: {'✅ Trained' if journey_summary.get('models_trained', {}).get('duration_model', False) else '❌ Not trained'}
            
            **POI Categories:**
            The system recognizes various POI types including restaurants, cafes, shops, supermarkets, 
            banks, ATMs, schools, parks, hospitals, pharmacies, and transportation hubs. Each POI is 
            scored for popularity and accessibility based on population density and proximity to bike stations.
            
            **Journey Types:**
            - **Direct**: No intermediate stops
            - **Errands**: Bank, shop, or service visits
            - **Lunch/Dining**: Meal-related stops
            - **Leisure**: Parks, entertainment venues
            - **Work**: Office or education-related
            - **Health**: Medical facilities
            """)
    
    # Legend for journey visualization
    with st.expander("🎨 Journey Visualization Legend", expanded=False):
        st.markdown("""
        ## Journey Type Icons
        
        - 🎯 **Direct**: Straight route with no stops
        - 🛍️ **Errands**: Shopping, banking, services
        - 🍽️ **Lunch**: Midday meal stops
        - 🍷 **Dining**: Evening dining experiences
        - 🎨 **Leisure**: Parks, museums, entertainment
        - 💼 **Work**: Office or business-related
        - 🏥 **Health**: Medical appointments
        
        ## POI Icons
        
        - 🍽️ Restaurants  - ☕ Cafes  - 🛍️ Shops  - 🛒 Supermarkets
        - 🏦 Banks  - 💳 ATMs  - 🏫 Schools  - 🌳 Parks
        - 🏥 Hospitals  - 💊 Pharmacies  - 🚌 Transit
        
        ## Confidence Levels
        
        - 🟢 **High (70%+)**: Strong prediction confidence
        - 🟡 **Medium (40-70%)**: Moderate confidence
        - 🔴 **Low (<40%)**: Lower confidence, explore with caution
        """)

if __name__ == "__main__":
    main()