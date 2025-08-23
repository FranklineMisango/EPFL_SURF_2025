"""
Main Cross-City Transfer Learning Application
Integrated system for spatial flow prediction across cities with interactive OSM interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
import sys
sys.path.append('..')

# Import configuration and utilities
from config import *
from utils import sanitize_for_logging, validate_coordinates, parse_coordinates_safe, get_confidence_class
from flowmap_integration import FlowmapVisualizer, integrate_with_existing_predictions



# Import always-required modules
from helpers.interactive_city_database import InteractiveCityDatabase

# Set optional modules to None by default
CrossCityTransferPredictor = None
TransferConfig = None
GNNConfig = None
create_enhanced_transfer_map = None
EnhancedOSMRouter = None
InteractiveFlowMap = None
OSMFeatureExtractor = None
PopulationFeatureExtractor = None

# Import optional enhanced modules
try:
    from cross_city_transfer_predictor import CrossCityTransferPredictor, TransferConfig, GNNConfig
    from enhanced_osm_integration import create_enhanced_transfer_map, EnhancedOSMRouter, InteractiveFlowMap
    from osm_feature_extractor import OSMFeatureExtractor
    from population_feature_extractor import PopulationFeatureExtractor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Some modules may not be available. The app will work with limited functionality.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Cross-City Spatial Flow Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for the integrated application
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .city-selection-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .transfer-status-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.3);
    }
    
    .prediction-result-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(252, 182, 159, 0.3);
        border-left: 5px solid #ff7f0e;
    }
    
    .metric-highlight {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .interactive-guide {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px dashed #d946ef;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


class CrossCityFlowApp:
    """Integrated Cross-City Transfer Learning and Flow Prediction Application"""
    
    def __init__(self):
        """Initialize the application"""
        self.city_db = InteractiveCityDatabase()
        self.osm_router = EnhancedOSMRouter() if EnhancedOSMRouter is not None else None
        self.flowmap_visualizer = FlowmapVisualizer()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        self._initialize_session_state()
        
        # Initialize predictor and other components
        self.transfer_predictor = None
        self.osm_extractor = OSMFeatureExtractor() if OSMFeatureExtractor is not None else None
        self.population_extractor = PopulationFeatureExtractor() if PopulationFeatureExtractor is not None else None
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'transfer_initialized' not in st.session_state:
            st.session_state.transfer_initialized = False
        if 'selected_stations' not in st.session_state:
            st.session_state.selected_stations = []
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'current_prediction' not in st.session_state:
            st.session_state.current_prediction = None
        if 'citywide_analysis' not in st.session_state:
            st.session_state.citywide_analysis = None
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🌍 Cross-City Spatial Flow Prediction</h1>', unsafe_allow_html=True)
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_interactive_map()
        
        with col2:
            self._render_prediction_panel()
        
        # Analysis dashboard at the bottom
        self._render_analysis_dashboard()
        
        # Flowmap.gl visualization
        self._render_flowmap_visualization()
    
    def _render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.markdown("## 🎛️ System Configuration")
            
            # City selection
            self._render_city_selection()
            
            # Transfer learning configuration
            self._render_transfer_config()
            
            # Temporal settings
            self._render_temporal_settings()
            
            # Action buttons
            self._render_action_buttons()
    
    def _render_city_selection(self):
        """Render city selection interface with multi-select for training cities"""
        st.markdown("""
        <div class="city-selection-card">
            <h3>🏙️ City Selection</h3>
        </div>
        """, unsafe_allow_html=True)

        available_cities = self.city_db.get_available_cities()

        # Multi-select for training cities
        source_cities = st.multiselect(
            "Training Cities (select one or more):",
            available_cities,
            default=[available_cities[0]] if available_cities else [],
            key="source_cities",
            help="Cities to use for model training (multi-select allowed)"
        )

        # Target city selection
        target_city = st.selectbox(
            "Target City (for prediction):",
            available_cities,
            index=1 if len(available_cities) > 1 else 0,
            key="target_city",
            help="City to apply transfer learning and make predictions"
        )

        # Display city information
        if source_cities and target_city:
            st.markdown(f"**📍 Training Cities:** {', '.join(source_cities)}")
            target_info = self.city_db.get_city_info(target_city)
            st.markdown(f"**🎯 Target:** {target_info['name']} ({target_info['stations']} stations)")
    
    def _render_transfer_config(self):
        """Render transfer learning configuration"""
        st.markdown("### 🔧 Transfer Learning Setup")
        
        # Model architecture selection
        architecture = st.selectbox(
            "GNN Architecture:",
            ["GCN", "GAT", "GraphSAGE"],
            index=1,
            help="Graph Neural Network architecture"
        )
        
        # Training parameters
        with st.expander("⚙️ Advanced Settings", expanded=False):
            hidden_dim = st.slider("Hidden Dimensions", 32, 256, 128)
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            epochs = st.slider("Training Epochs", 10, 200, 50)
            
        # Store configuration
        st.session_state.transfer_config = {
            'architecture': architecture,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'epochs': epochs
        }
    
    def _render_temporal_settings(self):
        """Render temporal configuration settings"""
        st.markdown("### ⏰ Temporal Settings")
        
        # Time of day
        hour = st.slider(
            "Hour of Day:",
            0, 23, 17,
            key="selected_hour",
            help="Hour for flow prediction (0-23)"
        )
        
        # Day of week
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week = st.selectbox(
            "Day of Week:",
            range(7),
            index=1,
            format_func=lambda x: day_names[x],
            key="selected_day_of_week",
            help="Day of week for prediction"
        )
        
        st.markdown(f"**Current Setting:** {hour:02d}:00 on {day_names[day_of_week]}")
    
    def _render_action_buttons(self):
        """Render action buttons for system operations"""
        st.markdown("### 🚀 Actions")
        
        # Initialize transfer system
        if st.button("🔄 Initialize Transfer System", use_container_width=True):
            self._initialize_transfer_system()
        
        # Status indicator
        if st.session_state.transfer_initialized:
            st.markdown("""
            <div class="transfer-status-card">
                <p><strong>✅ Transfer System Ready</strong></p>
                <p>Click stations on map to predict flows</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Random Prediction", use_container_width=True):
                self._make_random_prediction()
        
        with col2:
            if st.button("🌐 City-wide Analysis", use_container_width=True):
                self._perform_citywide_analysis()
    
    def _render_interactive_map(self):
        """Render the interactive map with clickable stations"""
        st.markdown("### 🗺️ Interactive Transfer Learning Map")
        
        # Instructions
        if not st.session_state.transfer_initialized:
            st.markdown("""
            <div class="interactive-guide">
                <h4>🎯 Getting Started</h4>
                <ol>
                    <li>Configure cities in sidebar</li>
                    <li>Click "Initialize Transfer System"</li>
                    <li>Click stations on map to predict flows</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Create enhanced map
        target_city = st.session_state.get('target_city', '')
        if target_city:
            stations = self.city_db.get_city_stations(target_city)
            city_center = next(iter(stations.values())) if stations else [46.9481, 7.4474]
            
            # Create the map
            m = create_enhanced_transfer_map(
                center_coords=city_center,
                stations=stations,
                selected_stations=st.session_state.selected_stations
            )
            
            # Display map and handle clicks
            map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])
            
            # Handle station clicks
            self._handle_map_clicks(map_data, stations)
    
    def _handle_map_clicks(self, map_data: Dict, stations: Dict):
        """Handle clicks on the interactive map"""
        if map_data["last_object_clicked"]:
            clicked_obj = map_data["last_object_clicked"]
            
            # Extract station name from popup or properties
            if "popup" in clicked_obj:
                # Parse station name from popup HTML
                popup_html = clicked_obj["popup"]
                if "Station:" in popup_html:
                    station_name = popup_html.split("Station:")[1].split("<")[0].strip()
                    
                    # Add to selection
                    if station_name not in st.session_state.selected_stations:
                        st.session_state.selected_stations.append(station_name)
                        
                        # Limit to 2 stations for flow prediction
                        if len(st.session_state.selected_stations) > 2:
                            st.session_state.selected_stations = st.session_state.selected_stations[-2:]
                        
                        # Trigger prediction if we have 2 stations
                        if len(st.session_state.selected_stations) == 2:
                            self._predict_flow_for_selection()
                        
                        st.rerun()
    
    def _initialize_transfer_system(self):
        """Initialize the transfer learning system with multiple training cities support"""
        source_cities = st.session_state.get('source_cities', [])
        target_city = st.session_state.get('target_city', '')

        if not source_cities or not target_city:
            st.error("Please select at least one training city and a target city.")
            return

        if target_city in source_cities:
            st.warning("Target city should not be among the training cities.")
            return

        with st.spinner("🔄 Initializing transfer learning system..."):
            try:
                # Create configurations
                config = st.session_state.get('transfer_config', {})

                # Create GNN configuration
                gnn_config = GNNConfig(
                    gnn_type=config.get('architecture', 'GAT'),
                    hidden_dim=config.get('hidden_dim', 128),
                    learning_rate=config.get('learning_rate', 0.001),
                    epochs=config.get('epochs', 50)
                )

                # Create transfer configuration
                transfer_config = TransferConfig(
                    source_city=','.join(source_cities),
                    target_city=target_city,
                    fine_tune_epochs=config.get('fine_tune_epochs', 50),
                    adaptation_weight=config.get('adaptation_weight', 0.1),
                    domain_adaptation=config.get('domain_adaptation', True)
                )

                # Initialize predictor
                self.transfer_predictor = CrossCityTransferPredictor(gnn_config, transfer_config)

                # Aggregate training data from all selected source cities
                all_source_trips = []
                all_source_stations = {}
                for city in source_cities:
                    trips, _ = self._generate_mock_training_data(city, target_city)
                    all_source_trips.append(trips)
                    stations = self.city_db.get_city_stations(city)
                    all_source_stations.update(stations)
                # Concatenate all trips
                import pandas as pd
                source_trips = pd.concat(all_source_trips, ignore_index=True) if all_source_trips else None

                # Target city data
                target_stations = self.city_db.get_city_stations(target_city)

                # Initialize transfer learning sequence
                self.transfer_predictor.load_source_city_data(source_trips, all_source_stations)
                self.transfer_predictor.train_source_model()
                self.transfer_predictor.prepare_target_city(target_stations, target_city)
                self.transfer_predictor.transfer_to_target_city()

                st.session_state.transfer_initialized = True
                st.success("✅ Transfer system initialized successfully!")
                logger.info(f"Transfer system initialized: {source_cities} → {target_city}")

            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                logger.error(f"Transfer initialization error: {e}")
    
    def _generate_mock_training_data(self, source_city: str, target_city: str):
        """Load real training data instead of generating mock data"""
        try:
            # Try to load real trip data first
            trips_file = "/Users/misango/codechest/EPFL_SURF_2025/data/trips_8days_flat.csv"
            stations_file = "/Users/misango/codechest/EPFL_SURF_2025/data/unique_stations.csv"
            
            if os.path.exists(trips_file) and os.path.exists(stations_file):
                # Load real data
                trips_df = pd.read_csv(trips_file)
                stations_df = pd.read_csv(stations_file)
                
                self.logger.info(f"🎯 Loading REAL data: {len(trips_df)} trips, {len(stations_df)} stations")
                
                # Split data for source and target cities (simulate different cities)
                # Use 80% for source city training, 20% for target city
                split_idx = int(len(trips_df) * 0.8)
                source_df = trips_df[:split_idx].copy()
                target_df = trips_df[split_idx:].copy()
                
                # Ensure required columns exist
                required_cols = ['start_station_id', 'end_station_id', 'hour', 'day_of_week']
                missing_cols = [col for col in required_cols if col not in source_df.columns]
                
                if missing_cols:
                    self.logger.warning(f"Missing columns {missing_cols}, adding defaults")
                    if 'hour' not in source_df.columns:
                        source_df['hour'] = np.random.choice([8, 17], size=len(source_df))
                        target_df['hour'] = np.random.choice([8, 17], size=len(target_df))
                    if 'day_of_week' not in source_df.columns:
                        source_df['day_of_week'] = np.random.choice(range(7), size=len(source_df))
                        target_df['day_of_week'] = np.random.choice(range(7), size=len(target_df))
                
                # Store in session state for the predictor to use
                st.session_state.source_training_data = source_df
                st.session_state.target_training_data = target_df
                st.session_state.real_stations = stations_df
                
                self.logger.info(f"✅ Using REAL data: {len(source_df)} source trips, {len(target_df)} target trips")
                return source_df, target_df
                
            else:
                self.logger.warning("⚠️ Real trip data not found, falling back to minimal mock data")
                return self._create_minimal_mock_data(source_city, target_city)
                
        except Exception as e:
            self.logger.error(f"❌ Error loading real data: {e}")
            return self._create_minimal_mock_data(source_city, target_city)
    
    def _create_minimal_mock_data(self, source_city: str, target_city: str):
        """Create minimal mock data as fallback"""
        np.random.seed(42)  # For reproducible results
        
        # Get stations for both cities
        source_stations = self.city_db.get_city_stations(source_city)
        target_stations = self.city_db.get_city_stations(target_city)
        
        # Generate minimal trips for source city
        source_trips = []
        for i in range(100):  # Much smaller dataset
            stations = list(source_stations.keys())
            start_station = np.random.choice(stations)
            end_station = np.random.choice([s for s in stations if s != start_station])
            
            source_trips.append({
                'trip_id': f"{source_city.lower()}_{i:03d}",
                'start_station_id': start_station,
                'end_station_id': end_station,
                'hour': np.random.choice([8, 17]),  # Peak hours only
                'day_of_week': np.random.choice([1, 2, 3, 4, 5])  # Weekdays only
            })
        
        # Generate minimal trips for target city
        target_trips = []
        for i in range(50):  # Even smaller for target
            stations = list(target_stations.keys())
            start_station = np.random.choice(stations)
            end_station = np.random.choice([s for s in stations if s != start_station])
            
            target_trips.append({
                'trip_id': f"{target_city.lower()}_{i:03d}",
                'start_station_id': start_station,
                'end_station_id': end_station,
                'hour': np.random.choice([8, 17]),
                'day_of_week': np.random.choice([1, 2, 3, 4, 5])
            })
        
        source_df = pd.DataFrame(source_trips)
        target_df = pd.DataFrame(target_trips)
        
        # Store in session state for the predictor to use
        st.session_state.source_training_data = source_df
        st.session_state.target_training_data = target_df
        
        self.logger.info(f"📝 Generated minimal mock data: {len(source_trips)} source trips, {len(target_trips)} target trips")
        
        return source_df, target_df
    
    def _predict_flow_for_selection(self):
        """Predict flow for the currently selected stations"""
        if not st.session_state.transfer_initialized or not self.transfer_predictor:
            st.error("Transfer system not initialized.")
            return
        
        source_station, dest_station = st.session_state.selected_stations
        hour = st.session_state.get('selected_hour', 17)
        day_of_week = st.session_state.get('selected_day_of_week', 1)
        
        try:
            with st.spinner("🔮 Predicting spatial flow..."):
                result = self.transfer_predictor.predict_target_flows(
                    source_station, dest_station, hour, day_of_week
                )
            # If result is a tuple, fallback to old style
            if isinstance(result, tuple):
                predicted_flow, confidence = result
                prediction_result = {
                    'source': source_station,
                    'target': dest_station,
                    'predicted_flow': predicted_flow,
                    'confidence': confidence,
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'timestamp': datetime.now()
                }
            else:
                prediction_result = {
                    'source': source_station,
                    'target': dest_station,
                    'predicted_flow': result.get('predicted_flow', 0.0),
                    'confidence': result.get('confidence', 0.0),
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'timestamp': datetime.now(),
                    'top_features': result.get('top_features', {}),
                    'all_features': result.get('all_features', [])
                }
            st.session_state.current_prediction = prediction_result
            st.session_state.prediction_history.append(prediction_result)
            st.success(f"✨ Prediction complete! Flow: {prediction_result['predicted_flow']:.2f} trips (Confidence: {prediction_result['confidence']:.1%})")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            logger.error(f"Prediction error: {e}")
    
    def _make_random_prediction(self):
        """Make a random prediction for demonstration"""
        if not st.session_state.transfer_initialized:
            st.warning("Please initialize transfer system first.")
            return
        
        target_city = st.session_state.get('target_city', '')
        stations = list(self.city_db.get_city_stations(target_city).keys())
        
        if not stations:
            st.warning("Not enough stations for prediction.")
            return
        
        source_station = np.random.choice(stations)
        dest_station = np.random.choice([s for s in stations if s != source_station])
        
        st.session_state.selected_stations = [source_station, dest_station]
        self._predict_flow_for_selection()
        st.rerun()
    
    def _perform_citywide_analysis(self):
        """Perform comprehensive city-wide flow analysis"""
        if not st.session_state.transfer_initialized:
            st.warning("Please initialize transfer system first.")
            return
        
        hour = st.session_state.get('selected_hour', 17)
        day_of_week = st.session_state.get('selected_day_of_week', 1)
        
        with st.spinner("🌐 Analyzing city-wide flow patterns..."):
            try:
                predictions = self.transfer_predictor.get_all_target_predictions(
                    hour=hour, day_of_week=day_of_week, top_k=15
                )
                
                st.session_state.citywide_analysis = predictions
                st.success(f"✅ Analyzed {len(predictions)} station pair flows")
                
            except Exception as e:
                st.error(f"City-wide analysis failed: {str(e)}")
                logger.error(f"Citywide analysis error: {e}")
    
    def _render_prediction_panel(self):
        """Render the prediction results panel"""
        st.markdown("### 🔮 Prediction Dashboard")
        
        # Current selection status
        self._display_selection_status()
        
        # Latest prediction results
        self._display_prediction_results()
        
        # Prediction history summary
        self._display_prediction_history()
    
    def _display_selection_status(self):
        """Display current station selection status"""
        selected = st.session_state.get('selected_stations', [])
        
        if len(selected) == 0:
            st.info("👆 Click on a station to start prediction")
        elif len(selected) == 1:
            st.markdown(f"""
            <div class="metric-highlight">
                <p><strong>📍 Source Selected:</strong> {selected[0]}</p>
                <p><em>Click another station for destination</em></p>
            </div>
            """, unsafe_allow_html=True)
        elif len(selected) == 2:
            st.markdown(f"""
            <div class="metric-highlight">
                <p><strong>📍 Source:</strong> {selected[0]}</p>
                <p><strong>🎯 Destination:</strong> {selected[1]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _display_prediction_results(self):
        """Display latest prediction results"""
        current_pred = st.session_state.get('current_prediction')
        
        if not current_pred:
            return
        
        # Confidence styling
        conf = current_pred['confidence']
        if conf > 0.7:
            conf_class = 'confidence-high'
            conf_emoji = '🟢'
        elif conf > 0.4:
            conf_class = 'confidence-medium'
            conf_emoji = '🟡'
        else:
            conf_class = 'confidence-low'
            conf_emoji = '🔴'
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        st.markdown(f"""
        <div class="prediction-result-card">
            <h4>🎯 Latest Flow Prediction</h4>
            <p><strong>Route:</strong> {current_pred['source']} → {current_pred['target']}</p>
            <p><strong>Time:</strong> {current_pred['hour']:02d}:00 on {day_names[current_pred['day_of_week']]}</p>
            <p><strong>Predicted Flow:</strong> {current_pred['predicted_flow']:.2f} trips</p>
            <p><strong>Confidence:</strong> {conf_emoji} <span class="{conf_class}">{conf:.1%}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show top features influencing the result (if available)
        if 'top_features' in current_pred:
            st.markdown("**🔬 Top Features Influencing This Prediction:**")
            top_feats = current_pred['top_features']
            for feat, val in top_feats.items():
                st.markdown(f"- **{feat}**: {val:.3f}")

        # Show all features used for training (if available)
        if 'all_features' in current_pred:
            with st.expander("🧩 All Features Used for Training", expanded=False):
                st.markdown(", ".join(current_pred['all_features']))

        # Additional analysis button
        if st.button("📊 Detailed Route Analysis"):
            self._show_detailed_route_analysis(current_pred)
    
    def _display_prediction_history(self):
        """Display prediction history summary"""
        history = st.session_state.get('prediction_history', [])
        
        if len(history) > 0:
            st.markdown(f"**📈 Prediction History:** {len(history)} predictions made")
            
            # Quick stats
            flows = [p['predicted_flow'] for p in history[-10:]]  # Last 10
            if flows:
                avg_flow = np.mean(flows)
                max_flow = np.max(flows)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Flow", f"{avg_flow:.1f}", help="Average predicted flow (last 10)")
                with col2:
                    st.metric("Max Flow", f"{max_flow:.1f}", help="Maximum predicted flow (last 10)")
    
    def _show_detailed_route_analysis(self, prediction: Dict):
        """Show detailed analysis for a specific prediction"""
        st.markdown("#### 🛣️ Detailed Route Analysis")
        
        # Get route information
        target_city = st.session_state.get('target_city', '')
        stations = self.city_db.get_city_stations(target_city)
        
        source_coords = stations.get(prediction['source'])
        target_coords = stations.get(prediction['target'])
        
        if source_coords and target_coords:
            route_info = self.osm_router.get_route_between_stations(source_coords, target_coords)
            
            if route_info:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Distance", f"{route_info.distance_km:.2f} km")
                with col2:
                    st.metric("Duration", f"{route_info.duration_minutes:.0f} min")
                with col3:
                    st.metric("Speed", f"{route_info.distance_km / (route_info.duration_minutes/60):.1f} km/h")
                
                # Route characteristics
                st.markdown("**🗺️ Route Characteristics:**")
                st.markdown(f"""
                - **Type:** {route_info.route_type.title()}
                - **Coordinates:** {len(route_info.coordinates)} waypoints
                - **Start:** ({source_coords[0]:.4f}, {source_coords[1]:.4f})
                - **End:** ({target_coords[0]:.4f}, {target_coords[1]:.4f})
                """)
    
    def _render_analysis_dashboard(self):
        """Render comprehensive analysis dashboard"""
        st.markdown("---")
        
        # City-wide analysis results
        citywide = st.session_state.get('citywide_analysis')
        if citywide:
            self._display_citywide_analysis(citywide)
        
        # System performance and insights
        self._display_system_insights()
    
    def _display_citywide_analysis(self, predictions: List[Dict]):
        """Display city-wide analysis results"""
        st.markdown("### 🌐 City-wide Flow Analysis (Intra-city Flows)")

        if not predictions:
            return

        # Only keep flows where both source and target are in the target city
        target_city = st.session_state.get('target_city', '')
        stations = set(self.city_db.get_city_stations(target_city).keys())
        df = pd.DataFrame(predictions)
        if not df.empty:
            df = df[df['source'].isin(stations) & df['target'].isin(stations)]

        # Summary statistics with empty check
        if not df.empty:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Flows", len(df))
            with col2:
                st.metric("Avg Flow", f"{df['predicted_flow'].mean():.2f}")
            with col3:
                st.metric("Max Flow", f"{df['predicted_flow'].max():.2f}")
            with col4:
                st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
        else:
            st.warning("No flows found for analysis.")
            return

        # Top flows table
        st.markdown("**🏆 Top Predicted Flows:**")
        display_df = df[['source', 'target', 'predicted_flow', 'confidence']].copy()
        display_df.columns = ['Source Station', 'Target Station', 'Predicted Flow', 'Confidence']
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
        display_df['Predicted Flow'] = display_df['Predicted Flow'].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df.head(10), use_container_width=True)

        # Visualizations
        tab1, tab2 = st.tabs(["📊 Flow Distribution", "🎯 Confidence Analysis"])

        with tab1:
            fig_hist = px.histogram(
                df, x='predicted_flow',
                title="Distribution of Predicted Flows",
                labels={'predicted_flow': 'Predicted Flow (trips)', 'count': 'Number of Routes'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab2:
            fig_scatter = px.scatter(
                df, x='confidence', y='predicted_flow',
                title="Prediction Confidence vs Flow Magnitude",
                labels={'confidence': 'Confidence', 'predicted_flow': 'Predicted Flow (trips)'},
                color='predicted_flow',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def _display_system_insights(self):
        """Display system performance insights and information"""
        with st.expander("📈 System Performance & Insights", expanded=False):
            if st.session_state.transfer_initialized and self.transfer_predictor:
                summary = self.transfer_predictor.get_transfer_summary()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🌍 Transfer Learning Summary**")
                    st.markdown(f"""
                    - **Source City:** {summary.get('source_city', 'N/A').title()}
                    - **Target City:** {summary.get('target_city', 'N/A').title()}
                    - **Source Stations:** {summary.get('source_stations', 0)}
                    - **Target Stations:** {summary.get('target_stations', 0)}
                    """)
                
                with col2:
                    st.markdown("**🤖 Model Status**")
                    models = summary.get('models_available', {})
                    st.markdown(f"""
                    - **Source Model:** {'✅ Trained' if models.get('source_model') else '❌ Not available'}
                    - **Transfer Model:** {'✅ Ready' if models.get('transfer_model') else '❌ Not available'}
                    - **Features:** OSM + Population + Temporal
                    - **Architecture:** Graph Neural Network
                    """)
            
            # Feature explanation
            st.markdown("**🔬 Technical Features**")
            st.markdown("""
            - **🧠 Graph Neural Networks:** Learn spatial relationships between stations
            - **🗺️ OpenStreetMap Integration:** Real-time amenity and infrastructure data
            - **👥 Population Analysis:** Demographic density and distribution
            - **🔄 Domain Adaptation:** Reduces bias between different cities
            - **⏰ Temporal Modeling:** Hour-of-day and day-of-week patterns
            - **🛣️ Route Optimization:** Multiple path alternatives with OSRM
            """)
    
    def _render_flowmap_visualization(self):
        """Render Flowmap.gl visualization for current predictions"""
        st.markdown("---")
        st.markdown("### 🌊 Interactive Flow Visualization (Flowmap.gl)")
        
        # Check if we have predictions to visualize
        citywide = st.session_state.get('citywide_analysis')
        current_pred = st.session_state.get('current_prediction')
        
        if not citywide and not current_pred:
            st.info("🎯 Generate predictions first to see Flowmap.gl visualization")
            return
        
        # Prepare data for visualization
        target_city = st.session_state.get('target_city', '')
        stations = self.city_db.get_city_stations(target_city)
        
        # Use citywide analysis if available, otherwise current prediction
        flows_to_visualize = []
        
        if citywide:
            flows_to_visualize = citywide
            viz_title = f"City-wide Flow Analysis - {target_city}"
        elif current_pred:
            flows_to_visualize = [{
                'source': current_pred['source'],
                'target': current_pred['target'],
                'predicted_flow': current_pred['predicted_flow']
            }]
            viz_title = f"Single Flow Prediction - {target_city}"
        
        if flows_to_visualize:
            # Generate Flowmap.gl HTML
            html_content = integrate_with_existing_predictions(flows_to_visualize, stations)
            
            # Display the visualization
            st.components.v1.html(html_content, height=600)
            
            # Show summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Flows Visualized", len(flows_to_visualize))
            with col2:
                total_flow = sum(f['predicted_flow'] for f in flows_to_visualize)
                st.metric("Total Flow", f"{total_flow:.1f}")
            with col3:
                avg_flow = total_flow / len(flows_to_visualize) if flows_to_visualize else 0
                st.metric("Average Flow", f"{avg_flow:.1f}")
            
            st.markdown("""
            **🌊 Flowmap.gl Features:**
            - **Animated Flows:** Watch flows move between stations in real-time
            - **Clustering:** Automatically groups nearby stations for better visualization
            - **Interactive:** Click on flows and stations for details
            - **Adaptive Scaling:** Flow thickness represents prediction magnitude
            """)

def main():
    """Main application entry point"""
    app = CrossCityFlowApp()
    app.run()

if __name__ == "__main__":
    main()
