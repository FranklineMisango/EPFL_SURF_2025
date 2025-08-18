"""
Interactive Cross-City Transfer Learning Application
Click-to-predict spatial flows with transfer learning between cities
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium, folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
import logging
from typing import Dict, List, Tuple, Optional
import requests
import zipfile
import io
import os
import math

# Import our enhanced modules
from cross_city_transfer_predictor import CrossCityTransferPredictor, TransferConfig, GNNConfig
from osm_feature_extractor import OSMFeatureExtractor

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    return c * r

class InteractiveCityDatabase:
    """Database of cities with station coordinates"""
    
    def __init__(self):
        # Swiss cities with stations and center coordinates
        self.cities_data = {
            'bern': {
                'center': (46.9481, 7.4474),
                'stations': {
                    'BE001': (46.9511, 7.4386),
                    'BE002': (46.9462, 7.4481),
                    'BE003': (46.9525, 7.4529),
                    'BE004': (46.9441, 7.4423),
                    'BE005': (46.9498, 7.4551),
                    'BE006': (46.9387, 7.4516),
                    'BE007': (46.9567, 7.4478),
                    'BE008': (46.9455, 7.4372),
                    'BE009': (46.9533, 7.4402),
                    'BE010': (46.9421, 7.4495)
                }
            },
            'zurich': {
                'center': (47.3769, 8.5417),
                'stations': {
                    'ZH001': (47.3776, 8.5406),
                    'ZH002': (47.3751, 8.5449),
                    'ZH003': (47.3789, 8.5388),
                    'ZH004': (47.3743, 8.5378),
                    'ZH005': (47.3799, 8.5456),
                    'ZH006': (47.3731, 8.5423),
                    'ZH007': (47.3812, 8.5381),
                    'ZH008': (47.3758, 8.5467),
                    'ZH009': (47.3784, 8.5412),
                    'ZH010': (47.3767, 8.5378)
                }
            },
            'basel': {
                'center': (47.5596, 7.5886),
                'stations': {
                    'BS001': (47.5584, 7.5899),
                    'BS002': (47.5612, 7.5871),
                    'BS003': (47.5578, 7.5923),
                    'BS004': (47.5623, 7.5845),
                    'BS005': (47.5567, 7.5903),
                    'BS006': (47.5601, 7.5936),
                    'BS007': (47.5634, 7.5889),
                    'BS008': (47.5589, 7.5854),
                    'BS009': (47.5611, 7.5912),
                    'BS010': (47.5576, 7.5867)
                }
            },
            'geneva': {
                'center': (46.2044, 6.1432),
                'stations': {
                    'GE001': (46.2051, 6.1419),
                    'GE002': (46.2033, 6.1456),
                    'GE003': (46.2067, 6.1403),
                    'GE004': (46.2021, 6.1445),
                    'GE005': (46.2078, 6.1467),
                    'GE006': (46.2015, 6.1398),
                    'GE007': (46.2089, 6.1423),
                    'GE008': (46.2038, 6.1489),
                    'GE009': (46.2074, 6.1412),
                    'GE010': (46.2027, 6.1434)
                }
            },
            'lausanne': {
                'center': (46.5197, 6.6323),
                'stations': {
                    'LA001': (46.5184, 6.6337),
                    'LA002': (46.5211, 6.6298),
                    'LA003': (46.5169, 6.6355),
                    'LA004': (46.5226, 6.6281),
                    'LA005': (46.5203, 6.6389),
                    'LA006': (46.5157, 6.6312),
                    'LA007': (46.5238, 6.6334),
                    'LA008': (46.5192, 6.6267),
                    'LA009': (46.5215, 6.6356),
                    'LA010': (46.5176, 6.6289)
                }
            }
        }
    
    def get_available_cities(self) -> List[str]:
        """Get list of available city names"""
        return list(self.cities_data.keys())
    
    def get_city_stations(self, city: str) -> Dict[str, Tuple[float, float]]:
        """Get stations for a specific city"""
        return self.cities_data.get(city.lower(), {}).get('stations', {})
    
    def get_city_metadata(self, city: str) -> Dict:
        """Get metadata for a specific city"""
        return self.cities_data.get(city.lower(), {})
    
    def get_city_info(self, city: str) -> Dict:
        """Get city information including name and station count"""
        city_data = self.cities_data.get(city.lower(), {})
        stations = city_data.get('stations', {})
        return {
            'name': city.title(),
            'stations': len(stations),
            'center': city_data.get('center', (0, 0))
        }

class InteractiveTransferApp:
    """Main Streamlit application for interactive transfer learning"""
    
    def __init__(self):
        self.city_db = InteractiveCityDatabase()
        self.transfer_predictor = None
        self.transfer_status = "Not initialized"
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configure page
        st.set_page_config(
            page_title="Cross-City Transfer Learning",
            page_icon="🚲",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._inject_custom_css()
    
    def _inject_custom_css(self):
        """Inject custom CSS for better styling"""
        st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .main-header {
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 1rem;
        }
        .transfer-status {
            background-color: rgba(173, 216, 230, 0.3);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #add8e6;
            margin: 0.5rem 0;
        }
        .prediction-card {
            background-color: rgba(255, 248, 225, 0.9);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            margin: 0.5rem 0;
        }
        .confidence-high { color: #4caf50; font-weight: bold; }
        .confidence-medium { color: #ff9800; font-weight: bold; }
        .confidence-low { color: #f44336; font-weight: bold; }
        .interactive-hint {
            background-color: rgba(255, 235, 59, 0.2);
            padding: 0.8rem;
            border-radius: 6px;
            border: 1px dashed #ffc107;
            margin: 0.5rem 0;
            font-style: italic;
        }
        .city-selector {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown('<h1 class="main-header">🚲 Cross-City Transfer Learning</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666;">Interactive spatial flow prediction with transfer learning</p>', unsafe_allow_html=True)
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_main_map()
        
        with col2:
            self._render_prediction_panel()
        
        # Sidebar for configuration
        self._render_sidebar()
        
        # Bottom analysis section
        self._render_analysis_section()
    
    def _render_sidebar(self):
        """Render sidebar with configuration options"""
        with st.sidebar:
            st.header("🎛️ Transfer Learning Configuration")
            
            # City selection
            st.markdown('<div class="city-selector">', unsafe_allow_html=True)
            st.subheader("🏙️ City Selection")
            
            available_cities = self.city_db.get_available_cities()
            
            source_city = st.selectbox(
                "Source City (training data)",
                available_cities,
                index=0,
                help="City with historical trip data for training"
            )
            
            target_city = st.selectbox(
                "Target City (prediction)",
                available_cities,
                index=1 if len(available_cities) > 1 else 0,
                help="City where predictions will be made"
            )
            
            st.session_state.source_city = source_city
            st.session_state.target_city = target_city
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Temporal configuration
            st.subheader("⏰ Temporal Configuration")
            
            selected_hour = st.slider(
                "Hour of day",
                min_value=0,
                max_value=23,
                value=17,
                help="Hour for prediction (0-23)"
            )
            
            selected_day_of_week = st.selectbox(
                "Day of week",
                options=list(range(7)),
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
                index=1,
                help="Day of week for prediction"
            )
            
            st.session_state.selected_hour = selected_hour
            st.session_state.selected_day_of_week = selected_day_of_week
            
            # Model configuration
            st.subheader("🧠 Model Configuration")
            
            gnn_type = st.selectbox(
                "GNN Architecture",
                ["GCN", "GAT", "GraphSAGE"],
                index=1,
                help="Graph Neural Network architecture"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                hidden_dim = st.selectbox("Hidden Dim", [64, 128, 256], index=1)
                num_layers = st.selectbox("Layers", [2, 3, 4], index=1)
            
            with col2:
                dropout = st.selectbox("Dropout", [0.1, 0.2, 0.3], index=1)
                fine_tune_epochs = st.selectbox("Fine-tune Epochs", [20, 50, 100], index=1)
            
            # Transfer learning configuration
            st.subheader("🔄 Transfer Configuration")
            
            adaptation_weight = st.slider(
                "Domain Adaptation Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Weight for domain adaptation loss"
            )
            
            use_domain_adaptation = st.checkbox(
                "Enable Domain Adaptation",
                value=True,
                help="Use domain adaptation for better transfer"
            )
            
            # Initialize transfer learning
            st.markdown("---")
            
            if st.button("🚀 Initialize Transfer Learning", type="primary"):
                if source_city == target_city:
                    st.warning("Please select different source and target cities")
                else:
                    self._initialize_transfer_learning(
                        source_city, target_city, gnn_type, hidden_dim, 
                        num_layers, dropout, fine_tune_epochs, 
                        adaptation_weight, use_domain_adaptation
                    )
            
            # Transfer status
            st.markdown("### 📊 Transfer Status")
            st.markdown(f'<div class="transfer-status">{self.transfer_status}</div>', unsafe_allow_html=True)
    
    def _initialize_transfer_learning(
        self, source_city: str, target_city: str, gnn_type: str,
        hidden_dim: int, num_layers: int, dropout: float,
        fine_tune_epochs: int, adaptation_weight: float,
        use_domain_adaptation: bool
    ):
        """Initialize the transfer learning system"""
        with st.spinner("Initializing transfer learning system..."):
            # Create configurations
            gnn_config = GNNConfig(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                gnn_type=gnn_type,
                epochs=100
            )
            
            transfer_config = TransferConfig(
                source_city=source_city.lower(),
                target_city=target_city.lower(),
                fine_tune_epochs=fine_tune_epochs,
                adaptation_weight=adaptation_weight,
                domain_adaptation=use_domain_adaptation
            )
            
            # Initialize predictor
            self.transfer_predictor = CrossCityTransferPredictor(gnn_config, transfer_config)
            
            # Load source city data (mock trip data)
            source_stations = self.city_db.get_city_stations(source_city)
            source_trips_df = self._generate_mock_trip_data(source_stations, source_city)
            
            # Train on source city
            status_text = st.empty()
            status_text.text("Loading source city data...")
            self.transfer_predictor.load_source_city_data(source_trips_df, source_stations)
            
            status_text.text("Training source model...")
            source_metrics = self.transfer_predictor.train_source_model()
            
            # Prepare target city
            status_text.text("Preparing target city...")
            target_stations = self.city_db.get_city_stations(target_city)
            self.transfer_predictor.prepare_target_city(target_stations, target_city)
            
            # Transfer to target
            status_text.text("Performing transfer learning...")
            transfer_metrics = self.transfer_predictor.transfer_to_target_city()
            
            status_text.empty()
            
            # Update status
            self.transfer_status = f"""
            <p><strong>✅ Source Model:</strong> R² = {source_metrics.get('r2', 0):.3f}</p>
            <p><strong>🔄 Transfer Type:</strong> {transfer_metrics.get('transfer_type', 'Unknown')}</p>
            <p><strong>🎯 Ready for Prediction</strong></p>
            """
            
            st.success(f"Transfer learning initialized! {source_city} → {target_city}")
    
    def _generate_mock_trip_data(self, stations: Dict[str, Tuple[float, float]], 
                               city_name: str) -> pd.DataFrame:
        """Generate mock trip data for source city training"""
        trips = []
        station_ids = list(stations.keys())
        
        # Generate realistic trip patterns
        for _ in range(1000):  # 1000 mock trips
            start_station = np.random.choice(station_ids)
            end_station = np.random.choice([s for s in station_ids if s != start_station])
            
            # Random temporal features
            hour = np.random.choice([7, 8, 9, 12, 13, 17, 18, 19], p=[0.1, 0.15, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15])
            day_of_week = np.random.choice(range(7))
            temperature = np.random.normal(15, 5)  # Mock temperature
            
            trips.append({
                'trip_id': f"{city_name.lower()}_{len(trips)}",
                'start_station_id': start_station,
                'end_station_id': end_station,
                'hour': hour,
                'day_of_week': day_of_week,
                'temperature': temperature,
                'start_coords': f"({stations[start_station][0]}, {stations[start_station][1]})",
                'end_coords': f"({stations[end_station][0]}, {stations[end_station][1]})"
            })
        
        return pd.DataFrame(trips)
    
    def _render_main_map(self):
        """Render the main interactive map"""
        if 'target_city' not in st.session_state:
            st.info("Please select cities and initialize the transfer system from the sidebar.")
            return
        
        target_city = st.session_state.get('target_city')
        target_stations = self.city_db.get_city_stations(target_city)
        
        if not target_stations:
            st.error(f"No station data available for {target_city}")
            return
        
        # Create interactive map
        center_lat, center_lon = self.city_db.get_city_metadata(target_city)['center']
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add station markers
        for station_id, (lat, lon) in target_stations.items():
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>Station {station_id}</b><br>Click to select for prediction",
                tooltip=f"Station {station_id}",
                icon=folium.Icon(color='blue', icon='bicycle', prefix='fa')
            ).add_to(m)
        
        # Add click instruction
        st.markdown("""
        <div class="interactive-hint">
            💡 <strong>Interactive Prediction:</strong> Click on any two stations on the map to predict flow between them!
            The first click selects the source station, the second click selects the destination.
        </div>
        """, unsafe_allow_html=True)
        
        # Display map and capture clicks
        map_data = st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"])
        
        # Handle map clicks
        if map_data['last_object_clicked']:
            self._handle_map_click(map_data['last_object_clicked'], target_stations)
    
    def _handle_map_click(self, click_data: Dict, stations: Dict[str, Tuple[float, float]]):
        """Handle map click for station selection"""
        if not click_data or 'lat' not in click_data or 'lng' not in click_data:
            return
        
        clicked_lat, clicked_lng = click_data['lat'], click_data['lng']
        
        # Find nearest station to click
        min_distance = float('inf')
        nearest_station = None
        
        for station_id, (lat, lon) in stations.items():
            distance = haversine_distance(clicked_lat, clicked_lng, lat, lon)
            if distance < min_distance and distance < 500:  # Within 500m
                min_distance = distance
                nearest_station = station_id
        
        if nearest_station:
            # Manage selected stations
            if 'selected_stations' not in st.session_state:
                st.session_state.selected_stations = []
            
            selected = st.session_state.selected_stations
            
            if len(selected) == 0:
                # First selection - source station
                selected.append(nearest_station)
                st.session_state.selected_stations = selected
                st.success(f"🎯 Source station selected: {nearest_station}")
                
            elif len(selected) == 1:
                # Second selection - destination station
                if nearest_station != selected[0]:
                    selected.append(nearest_station)
                    st.session_state.selected_stations = selected
                    st.success(f"🎯 Destination station selected: {nearest_station}")
                    
                    # Trigger prediction
                    self._predict_flow_for_selection()
                else:
                    st.warning("Please select a different station for destination.")
                    
            else:
                # Reset and start new selection
                st.session_state.selected_stations = [nearest_station]
                st.info(f"🔄 New selection started. Source: {nearest_station}")
    
    def _predict_flow_for_selection(self):
        """Predict flow for selected station pair"""
        if 'selected_stations' not in st.session_state or len(st.session_state.selected_stations) != 2:
            return
        
        if not self.transfer_predictor:
            st.error("Transfer system not initialized. Please initialize from sidebar.")
            return
        
        source_station, dest_station = st.session_state.selected_stations
        hour = st.session_state.get('selected_hour', 17)
        day_of_week = st.session_state.get('selected_day_of_week', 1)
        
        try:
            with st.spinner("Predicting flow..."):
                predicted_flow, confidence = self.transfer_predictor.predict_target_flows(
                    source_station, dest_station, hour, day_of_week
                )
            
            # Store prediction result
            st.session_state.latest_prediction = {
                'source': source_station,
                'destination': dest_station,
                'flow': predicted_flow,
                'confidence': confidence,
                'hour': hour,
                'day_of_week': day_of_week
            }
            
            st.success(f"✅ Prediction complete! Flow: {predicted_flow:.2f} trips")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    def _render_prediction_panel(self):
        """Render prediction results panel"""
        st.subheader("🔮 Prediction Results")
        
        # Current selection status
        if 'selected_stations' in st.session_state:
            selected = st.session_state.selected_stations
            if len(selected) == 1:
                st.markdown(f"""
                <div class="prediction-card">
                    <p><strong>📍 Source Selected:</strong> {selected[0]}</p>
                    <p><em>Click another station for destination</em></p>
                </div>
                """, unsafe_allow_html=True)
            elif len(selected) == 2:
                st.markdown(f"""
                <div class="prediction-card">
                    <p><strong>📍 Source:</strong> {selected[0]}</p>
                    <p><strong>🎯 Destination:</strong> {selected[1]}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Latest prediction results
        if 'latest_prediction' in st.session_state:
            pred = st.session_state.latest_prediction
            
            # Confidence styling
            if pred['confidence'] > 0.7:
                conf_class = 'confidence-high'
                conf_emoji = '🟢'
            elif pred['confidence'] > 0.4:
                conf_class = 'confidence-medium'
                conf_emoji = '🟡'
            else:
                conf_class = 'confidence-low'
                conf_emoji = '🔴'
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            st.markdown(f"""
            <div class="prediction-card">
                <h4>🎯 Flow Prediction</h4>
                <p><strong>Route:</strong> {pred['source']} → {pred['destination']}</p>
                <p><strong>Time:</strong> {pred['hour']:02d}:00 on {day_names[pred['day_of_week']]}</p>
                <p><strong>Predicted Flow:</strong> {pred['flow']:.2f} trips</p>
                <p><strong>Confidence:</strong> {conf_emoji} <span class="{conf_class}">{pred['confidence']:.1%}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional analysis
            if st.button("📊 Detailed Analysis"):
                self._show_detailed_analysis(pred)
        
        # Control buttons
        st.markdown("---")
        
        if st.button("🔄 Clear Selection"):
            if 'selected_stations' in st.session_state:
                del st.session_state.selected_stations
            st.rerun()
        
        if st.button("🎲 Random Prediction") and self.transfer_predictor:
            self._make_random_prediction()
        
        # Quick predictions for all station pairs
        if st.button("🌐 City-wide Analysis") and self.transfer_predictor:
            self._perform_citywide_analysis()
    
    def _make_random_prediction(self):
        """Make a random prediction for demonstration"""
        target_city = st.session_state.get('target_city')
        if not target_city:
            return
        
        stations = list(self.city_db.get_city_stations(target_city).keys())
        if len(stations) < 2:
            return
        
        source_station = np.random.choice(stations)
        dest_station = np.random.choice([s for s in stations if s != source_station])
        
        st.session_state.selected_stations = [source_station, dest_station]
        self._predict_flow_for_selection()
        st.rerun()
    
    def _perform_citywide_analysis(self):
        """Perform city-wide flow analysis"""
        hour = st.session_state.get('selected_hour', 17)
        day_of_week = st.session_state.get('selected_day_of_week', 1)
        
        with st.spinner("Analyzing city-wide flow patterns..."):
            predictions = self.transfer_predictor.get_all_target_predictions(
                hour=hour, day_of_week=day_of_week, top_k=10
            )
        
        st.session_state.citywide_predictions = predictions
        st.success(f"✅ Analyzed {len(predictions)} flow predictions")
    
    def _show_detailed_analysis(self, prediction: Dict):
        """Show detailed analysis for a prediction"""
        st.markdown("### 📈 Detailed Flow Analysis")
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Flow",
                f"{prediction['flow']:.2f}",
                help="Number of trips predicted between stations"
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{prediction['confidence']:.1%}",
                help="Model confidence in prediction"
            )
        
        with col3:
            flow_category = "High" if prediction['flow'] > 5 else "Medium" if prediction['flow'] > 2 else "Low"
            st.metric(
                "Flow Category",
                flow_category,
                help="Categorized flow level"
            )
        
        # Feature analysis (mock)
        st.markdown("**🗺️ Contributing Factors:**")
        st.markdown("""
        - **OSM Features:** High density of restaurants and shops near source station
        - **Population:** Dense residential area around destination
        - **Temporal:** Peak commuting hour with high expected mobility
        - **Transfer Learning:** Pattern similarity with source city data
        """)
    
    def _render_analysis_section(self):
        """Render bottom analysis section"""
        st.markdown("---")
        
        # City-wide analysis results
        if 'citywide_predictions' in st.session_state:
            predictions = st.session_state.citywide_predictions
            
            st.subheader("🌐 City-wide Flow Analysis")
            
            if predictions:
                # Create DataFrame for display
                df = pd.DataFrame(predictions)
                
                # Display top flows
                st.markdown("**Top Predicted Flows:**")
                display_df = df[['source', 'target', 'predicted_flow', 'confidence']].copy()
                display_df.columns = ['Source', 'Target', 'Predicted Flow', 'Confidence']
                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
                display_df['Predicted Flow'] = display_df['Predicted Flow'].apply(lambda x: f"{x:.2f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Flow distribution
                    fig_hist = px.histogram(
                        df, x='predicted_flow',
                        title="Distribution of Predicted Flows",
                        labels={'predicted_flow': 'Predicted Flow', 'count': 'Number of Station Pairs'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Confidence vs Flow
                    fig_scatter = px.scatter(
                        df, x='confidence', y='predicted_flow',
                        title="Prediction Confidence vs Flow",
                        labels={'confidence': 'Confidence', 'predicted_flow': 'Predicted Flow'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Transfer learning insights
        if self.transfer_predictor:
            summary = self.transfer_predictor.get_transfer_summary()
            
            with st.expander("🧠 Transfer Learning Summary", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Source City:** {summary.get('source_city', 'N/A').title()}
                    **Target City:** {summary.get('target_city', 'N/A').title()}
                    **Source Stations:** {summary.get('source_stations', 0)}
                    **Target Stations:** {summary.get('target_stations', 0)}
                    """)
                
                with col2:
                    models = summary.get('models_available', {})
                    st.markdown(f"""
                    **Source Model:** {'✅ Available' if models.get('source_model') else '❌ Not available'}
                    **Transfer Model:** {'✅ Available' if models.get('transfer_model') else '❌ Not available'}
                    **Transfer Type:** Domain adaptation enabled
                    **Features:** OSM + Population + Temporal
                    """)
        
        # System performance info
        with st.expander("ℹ️ System Information", expanded=False):
            st.markdown("""
            ### Cross-City Transfer Learning System
            
            **Features:**
            - 🧠 Graph Neural Network with transfer learning
            - 🗺️ OpenStreetMap feature extraction
            - 👥 Population density integration
            - 🔄 Domain adaptation for cross-city transfer
            - 📍 Interactive click-to-predict interface
            
            **Supported Cities:**
            - Bern, Zurich, Basel, Geneva, Lausanne (Switzerland)
            - Extensible to any city with coordinate data
            
            **Transfer Learning Process:**
            1. Train GNN on source city with historical trip data
            2. Extract OSM and population features for target city
            3. Align feature distributions between cities
            4. Apply domain adaptation to reduce city-specific bias
            5. Generate predictions for target city station pairs
            
            **Click-to-Predict Workflow:**
            1. Select source and target cities
            2. Initialize transfer learning system
            3. Click first station (source) on map
            4. Click second station (destination) on map
            5. View predicted flow and confidence
            """)

def main():
    """Main application entry point"""
    app = InteractiveTransferApp()
    app.run()

if __name__ == "__main__":
    main()
