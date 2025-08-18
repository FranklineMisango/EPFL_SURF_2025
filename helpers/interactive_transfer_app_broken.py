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
from population_feature_extractor import PopulationFeatureExtractor
from multi_path_router import ImprovedMultiPathRouter
import math

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Cross-City Flow Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for interactive features
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .city-card {
        background-color: rgba(240, 248, 255, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .transfer-status {
        background-color: rgba(232, 245, 233, 0.9);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
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
        background-color: rgba(225, 245, 254, 0.8);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class InteractiveCityDatabase:
    """Database of cities with their station data"""
    
    def __init__(self):
        self.cities = {}
        self.city_metadata = {}
        self._load_predefined_cities()
    
    def _load_predefined_cities(self):
        """Load predefined city data"""
        # Example cities with mock station data
        self.cities = {
            "Bern": {
                "stations": {
                    "bern_001": (46.9481, 7.4474),  # Bern Hauptbahnhof area
                    "bern_002": (46.9524, 7.4396),  # Old Town
                    "bern_003": (46.9570, 7.4474),  # University area
                    "bern_004": (46.9436, 7.4395),  # Wankdorf area
                    "bern_005": (46.9353, 7.4180),  # Köniz area
                    "bern_006": (46.9463, 7.4668),  # Ostermundigen
                    "bern_007": (46.9595, 7.4105),  # Bethlehem
                    "bern_008": (46.9282, 7.4470),  # Bümpliz
                },
                "country": "Switzerland",
                "population": 133000,
                "area_km2": 51.6,
                "has_bike_share": True
            },
            "Zurich": {
                "stations": {
                    "zur_001": (47.3769, 8.5417),  # Zurich HB
                    "zur_002": (47.3667, 8.5500),  # University area
                    "zur_003": (47.3886, 8.5164),  # Oerlikon
                    "zur_004": (47.3480, 8.5362),  # Enge
                    "zur_005": (47.4106, 8.5446),  # Schwamendingen
                    "zur_006": (47.3608, 8.5145),  # Wiedikon
                    "zur_007": (47.3915, 8.4729),  # Affoltern
                    "zur_008": (47.3382, 8.5267),  # Leimbach
                    "zur_009": (47.3789, 8.5088),  # Altstetten
                    "zur_010": (47.4058, 8.5663),  # Dübendorf
                },
                "country": "Switzerland",
                "population": 421000,
                "area_km2": 87.9,
                "has_bike_share": True
            },
            "Basel": {
                "stations": {
                    "bas_001": (47.5596, 7.5886),  # Basel SBB
                    "bas_002": (47.5584, 7.5733),  # Old Town
                    "bas_003": (47.5517, 7.5816),  # University
                    "bas_004": (47.5650, 7.5933),  # St. Johann
                    "bas_005": (47.5479, 7.5951),  # Gundeldingen
                    "bas_006": (47.5329, 7.6178),  # Birsfelden
                    "bas_007": (47.5744, 7.5678),  # Kleinbasel
                },
                "country": "Switzerland",
                "population": 177000,
                "area_km2": 23.9,
                "has_bike_share": True
            },
            "Geneva": {
                "stations": {
                    "gen_001": (46.2104, 6.1432),  # Geneva Cornavin
                    "gen_002": (46.2044, 6.1432),  # Old Town
                    "gen_003": (46.2276, 6.1372),  # UN area
                    "gen_004": (46.1983, 6.1587),  # Carouge
                    "gen_005": (46.1829, 6.1287),  # Plan-les-Ouates
                    "gen_006": (46.2200, 6.1095),  # Vernier
                    "gen_007": (46.1737, 6.1143),  # Lancy
                },
                "country": "Switzerland",
                "population": 203000,
                "area_km2": 15.9,
                "has_bike_share": True
            },
            "Lausanne": {
                "stations": {
                    "lau_001": (46.5197, 6.6323),  # Lausanne station
                    "lau_002": (46.5156, 6.6269),  # Old Town
                    "lau_003": (46.5238, 6.6356),  # Flon
                    "lau_004": (46.5352, 6.6024),  # EPFL area
                    "lau_005": (46.5025, 6.6476),  # Malley
                    "lau_006": (46.5472, 6.6590),  # Renens
                },
                "country": "Switzerland",
                "population": 140000,
                "area_km2": 41.4,
                "has_bike_share": True
            }
        }
        
        self.city_metadata = {
            city: {
                "station_count": len(data["stations"]),
                "bbox": self._calculate_bbox(data["stations"]),
                "center": self._calculate_center(data["stations"])
            }
            for city, data in self.cities.items()
        }
    
    def _calculate_bbox(self, stations: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Calculate bounding box for city stations"""
        lats = [coord[0] for coord in stations.values()]
        lons = [coord[1] for coord in stations.values()]
        return min(lats), min(lons), max(lats), max(lons)
    
    def _calculate_center(self, stations: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate center point of city stations"""
        lats = [coord[0] for coord in stations.values()]
        lons = [coord[1] for coord in stations.values()]
        return np.mean(lats), np.mean(lons)
    
    def get_city_list(self) -> List[str]:
        """Get list of available cities"""
        return list(self.cities.keys())
    
    def get_city_data(self, city_name: str) -> Dict:
        """Get complete city data"""
        return self.cities.get(city_name, {})
    
    def get_city_stations(self, city_name: str) -> Dict[str, Tuple[float, float]]:
        """Get station coordinates for a city"""
        city_data = self.cities.get(city_name, {})
        return city_data.get("stations", {})
    
    def get_city_metadata(self, city_name: str) -> Dict:
        """Get city metadata"""
        return self.city_metadata.get(city_name, {})

class InteractiveTransferApp:
    """Main interactive transfer learning application"""
    
    def __init__(self):
        self.city_db = InteractiveCityDatabase()
        self.transfer_predictor = None
        self.current_map_data = None
        self.selected_points = []
        
        # Initialize components
        self.osm_extractor = OSMFeatureExtractor()
        self.population_extractor = PopulationFeatureExtractor()
        self.router = ImprovedMultiPathRouter()
        
    def run(self):
        """Run the interactive application"""
        st.markdown('<h1 class="main-header">🌍 Cross-City Transfer Learning for Spatial Flow Prediction</h1>', unsafe_allow_html=True)
        
        # Sidebar configuration
        with st.sidebar:
            self._render_sidebar()
        
        # Main content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self._render_main_map()
        
        with col2:
            self._render_prediction_panel()
        
        # Bottom section - Analysis and results
        self._render_analysis_section()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.header("🎛️ Transfer Learning Controls")
        
        # City selection
        st.markdown('<div class="city-selector">', unsafe_allow_html=True)
        st.subheader("📍 City Selection")
        
        cities = self.city_db.get_city_list()
        
        source_city = st.selectbox(
            "Source City (Training Data)",
            cities,
            index=0,
            help="City with historical trip data for training"
        )
        
        target_city = st.selectbox(
            "Target City (Prediction)",
            cities,
            index=1 if len(cities) > 1 else 0,
            help="City where we want to predict flows"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display city information
        if source_city:
            source_data = self.city_db.get_city_data(source_city)
            st.markdown(f"""
            <div class="city-card">
                <h4>🎯 Source: {source_city}</h4>
                <p><strong>Stations:</strong> {len(source_data.get('stations', {}))}</p>
                <p><strong>Population:</strong> {source_data.get('population', 'N/A'):,}</p>
                <p><strong>Area:</strong> {source_data.get('area_km2', 'N/A')} km²</p>
            </div>
            """, unsafe_allow_html=True)
        
        if target_city and target_city != source_city:
            target_data = self.city_db.get_city_data(target_city)
            st.markdown(f"""
            <div class="city-card">
                <h4>🎯 Target: {target_city}</h4>
                <p><strong>Stations:</strong> {len(target_data.get('stations', {}))}</p>
                <p><strong>Population:</strong> {target_data.get('population', 'N/A'):,}</p>
                <p><strong>Area:</strong> {target_data.get('area_km2', 'N/A')} km²</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Transfer learning configuration
        st.markdown("---")
        st.subheader("🧠 Transfer Configuration")
        
        with st.expander("Model Configuration", expanded=False):
            gnn_type = st.selectbox("GNN Architecture", ["GCN", "GAT", "GraphSAGE"], index=0)
            hidden_dim = st.slider("Hidden Dimensions", 32, 256, 128, step=32)
            num_layers = st.slider("Number of Layers", 1, 5, 2)
            dropout = st.slider("Dropout Rate", 0.0, 0.8, 0.2, step=0.1)
        
        with st.expander("Transfer Settings", expanded=False):
            fine_tune_epochs = st.slider("Fine-tune Epochs", 10, 200, 50, step=10)
            adaptation_weight = st.slider("Domain Adaptation Weight", 0.0, 0.5, 0.1, step=0.05)
            use_domain_adaptation = st.checkbox("Enable Domain Adaptation", True)
        
        # Initialize transfer system
        if st.button("🚀 Initialize Transfer System", type="primary"):
            self._initialize_transfer_system(
                source_city, target_city, gnn_type, hidden_dim, 
                num_layers, dropout, fine_tune_epochs, adaptation_weight, use_domain_adaptation
            )
        
        # Transfer status
        if hasattr(self, 'transfer_status'):
            st.markdown(f"""
            <div class="transfer-status">
                <h5>📊 Transfer Status</h5>
                {self.transfer_status}
            </div>
            """, unsafe_allow_html=True)
        
        # Temporal controls
        st.markdown("---")
        st.subheader("⏰ Prediction Time")
        
        selected_hour = st.slider("Hour", 0, 23, 17)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        selected_day_name = st.selectbox("Day of Week", day_names, index=1)
        selected_day_of_week = day_names.index(selected_day_name)
        
        # Store selections in session state
        st.session_state.update({
            'source_city': source_city,
            'target_city': target_city,
            'selected_hour': selected_hour,
            'selected_day_of_week': selected_day_of_week
        })
    
    def _initialize_transfer_system(self, source_city: str, target_city: str, 
                                  gnn_type: str, hidden_dim: int, num_layers: int,
                                  dropout: float, fine_tune_epochs: int, 
                                  adaptation_weight: float, use_domain_adaptation: bool):
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
            
            # Initialize predictor\n            self.transfer_predictor = CrossCityTransferPredictor(gnn_config, transfer_config)\n            \n            # Load source city data (mock trip data)\n            source_stations = self.city_db.get_city_stations(source_city)\n            source_trips_df = self._generate_mock_trip_data(source_stations, source_city)\n            \n            # Train on source city\n            status_text = st.empty()\n            status_text.text("Loading source city data...")\n            self.transfer_predictor.load_source_city_data(source_trips_df, source_stations)\n            \n            status_text.text("Training source model...")\n            source_metrics = self.transfer_predictor.train_source_model()\n            \n            # Prepare target city\n            status_text.text("Preparing target city...")\n            target_stations = self.city_db.get_city_stations(target_city)\n            self.transfer_predictor.prepare_target_city(target_stations, target_city)\n            \n            # Transfer to target\n            status_text.text("Performing transfer learning...")\n            transfer_metrics = self.transfer_predictor.transfer_to_target_city()\n            \n            status_text.empty()\n            \n            # Update status\n            self.transfer_status = f\"\"\"\n            <p><strong>✅ Source Model:</strong> R² = {source_metrics.get('r2', 0):.3f}</p>\n            <p><strong>🔄 Transfer Type:</strong> {transfer_metrics.get('transfer_type', 'Unknown')}</p>\n            <p><strong>🎯 Ready for Prediction</strong></p>\n            \"\"\"\n            \n            st.success(f"Transfer learning initialized! {source_city} → {target_city}")\n    \n    def _generate_mock_trip_data(self, stations: Dict[str, Tuple[float, float]], \n                               city_name: str) -> pd.DataFrame:\n        \"\"\"Generate mock trip data for source city training\"\"\"\n        trips = []\n        station_ids = list(stations.keys())\n        \n        # Generate realistic trip patterns\n        for _ in range(1000):  # 1000 mock trips\n            start_station = np.random.choice(station_ids)\n            end_station = np.random.choice([s for s in station_ids if s != start_station])\n            \n            # Random temporal features\n            hour = np.random.choice([7, 8, 9, 12, 13, 17, 18, 19], p=[0.1, 0.15, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15])\n            day_of_week = np.random.choice(range(7))\n            temperature = np.random.normal(15, 5)  # Mock temperature\n            \n            trips.append({\n                'trip_id': f\"{city_name.lower()}_{len(trips)}\",\n                'start_station_id': start_station,\n                'end_station_id': end_station,\n                'hour': hour,\n                'day_of_week': day_of_week,\n                'temperature': temperature,\n                'start_coords': f\"({stations[start_station][0]}, {stations[start_station][1]})\",\n                'end_coords': f\"({stations[end_station][0]}, {stations[end_station][1]})\"\n            })\n        \n        return pd.DataFrame(trips)\n    \n    def _render_main_map(self):\n        \"\"\"Render the main interactive map\"\"\"\n        if 'target_city' not in st.session_state:\n            st.info(\"Please select cities and initialize the transfer system from the sidebar.\")\n            return\n        \n        target_city = st.session_state.get('target_city')\n        target_stations = self.city_db.get_city_stations(target_city)\n        \n        if not target_stations:\n            st.error(f\"No station data available for {target_city}\")\n            return\n        \n        # Create interactive map\n        center_lat, center_lon = self.city_db.get_city_metadata(target_city)['center']\n        \n        m = folium.Map(\n            location=[center_lat, center_lon],\n            zoom_start=12,\n            tiles='OpenStreetMap'\n        )\n        \n        # Add station markers\n        for station_id, (lat, lon) in target_stations.items():\n            folium.Marker(\n                location=[lat, lon],\n                popup=f\"<b>Station {station_id}</b><br>Click to select for prediction\",\n                tooltip=f\"Station {station_id}\",\n                icon=folium.Icon(color='blue', icon='bicycle', prefix='fa')\n            ).add_to(m)\n        \n        # Add click instruction\n        st.markdown(\"\"\"\n        <div class=\"interactive-hint\">\n            💡 <strong>Interactive Prediction:</strong> Click on any two stations on the map to predict flow between them!\n            The first click selects the source station, the second click selects the destination.\n        </div>\n        \"\"\", unsafe_allow_html=True)\n        \n        # Display map and capture clicks\n        map_data = st_folium(m, width=None, height=600, returned_objects=[\"last_object_clicked\"])\n        \n        # Handle map clicks\n        if map_data['last_object_clicked']:\n            self._handle_map_click(map_data['last_object_clicked'], target_stations)\n    \n    def _handle_map_click(self, click_data: Dict, stations: Dict[str, Tuple[float, float]]):\n        \"\"\"Handle map click for station selection\"\"\"\n        if not click_data or 'lat' not in click_data or 'lng' not in click_data:\n            return\n        \n        clicked_lat, clicked_lng = click_data['lat'], click_data['lng']\n        \n        # Find nearest station to click\n        min_distance = float('inf')\n        nearest_station = None\n        \n        for station_id, (lat, lon) in stations.items():\n            distance = geodesic((clicked_lat, clicked_lng), (lat, lon)).meters\n            if distance < min_distance and distance < 500:  # Within 500m\n                min_distance = distance\n                nearest_station = station_id\n        \n        if nearest_station:\n            # Manage selected stations\n            if 'selected_stations' not in st.session_state:\n                st.session_state.selected_stations = []\n            \n            selected = st.session_state.selected_stations\n            \n            if len(selected) == 0:\n                # First selection - source station\n                selected.append(nearest_station)\n                st.session_state.selected_stations = selected\n                st.success(f\"🎯 Source station selected: {nearest_station}\")\n                \n            elif len(selected) == 1:\n                # Second selection - destination station\n                if nearest_station != selected[0]:\n                    selected.append(nearest_station)\n                    st.session_state.selected_stations = selected\n                    st.success(f\"🎯 Destination station selected: {nearest_station}\")\n                    \n                    # Trigger prediction\n                    self._predict_flow_for_selection()\n                else:\n                    st.warning(\"Please select a different station for destination.\")\n                    \n            else:\n                # Reset and start new selection\n                st.session_state.selected_stations = [nearest_station]\n                st.info(f\"🔄 New selection started. Source: {nearest_station}\")\n    \n    def _predict_flow_for_selection(self):\n        \"\"\"Predict flow for selected station pair\"\"\"\n        if 'selected_stations' not in st.session_state or len(st.session_state.selected_stations) != 2:\n            return\n        \n        if not self.transfer_predictor:\n            st.error(\"Transfer system not initialized. Please initialize from sidebar.\")\n            return\n        \n        source_station, dest_station = st.session_state.selected_stations\n        hour = st.session_state.get('selected_hour', 17)\n        day_of_week = st.session_state.get('selected_day_of_week', 1)\n        \n        try:\n            with st.spinner(\"Predicting flow...\"):\n                predicted_flow, confidence = self.transfer_predictor.predict_target_flows(\n                    source_station, dest_station, hour, day_of_week\n                )\n            \n            # Store prediction result\n            st.session_state.latest_prediction = {\n                'source': source_station,\n                'destination': dest_station,\n                'flow': predicted_flow,\n                'confidence': confidence,\n                'hour': hour,\n                'day_of_week': day_of_week\n            }\n            \n            st.success(f\"✅ Prediction complete! Flow: {predicted_flow:.2f} trips\")\n            \n        except Exception as e:\n            st.error(f\"Prediction failed: {str(e)}\")\n    \n    def _render_prediction_panel(self):\n        \"\"\"Render prediction results panel\"\"\"\n        st.subheader(\"🔮 Prediction Results\")\n        \n        # Current selection status\n        if 'selected_stations' in st.session_state:\n            selected = st.session_state.selected_stations\n            if len(selected) == 1:\n                st.markdown(f\"\"\"\n                <div class=\"prediction-card\">\n                    <p><strong>📍 Source Selected:</strong> {selected[0]}</p>\n                    <p><em>Click another station for destination</em></p>\n                </div>\n                \"\"\", unsafe_allow_html=True)\n            elif len(selected) == 2:\n                st.markdown(f\"\"\"\n                <div class=\"prediction-card\">\n                    <p><strong>📍 Source:</strong> {selected[0]}</p>\n                    <p><strong>🎯 Destination:</strong> {selected[1]}</p>\n                </div>\n                \"\"\", unsafe_allow_html=True)\n        \n        # Latest prediction results\n        if 'latest_prediction' in st.session_state:\n            pred = st.session_state.latest_prediction\n            \n            # Confidence styling\n            if pred['confidence'] > 0.7:\n                conf_class = 'confidence-high'\n                conf_emoji = '🟢'\n            elif pred['confidence'] > 0.4:\n                conf_class = 'confidence-medium'\n                conf_emoji = '🟡'\n            else:\n                conf_class = 'confidence-low'\n                conf_emoji = '🔴'\n            \n            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n            \n            st.markdown(f\"\"\"\n            <div class=\"prediction-card\">\n                <h4>🎯 Flow Prediction</h4>\n                <p><strong>Route:</strong> {pred['source']} → {pred['destination']}</p>\n                <p><strong>Time:</strong> {pred['hour']:02d}:00 on {day_names[pred['day_of_week']]}</p>\n                <p><strong>Predicted Flow:</strong> {pred['flow']:.2f} trips</p>\n                <p><strong>Confidence:</strong> {conf_emoji} <span class=\"{conf_class}\">{pred['confidence']:.1%}</span></p>\n            </div>\n            \"\"\", unsafe_allow_html=True)\n            \n            # Additional analysis\n            if st.button(\"📊 Detailed Analysis\"):\n                self._show_detailed_analysis(pred)\n        \n        # Control buttons\n        st.markdown(\"---\")\n        \n        if st.button(\"🔄 Clear Selection\"):\n            if 'selected_stations' in st.session_state:\n                del st.session_state.selected_stations\n            st.rerun()\n        \n        if st.button(\"🎲 Random Prediction\") and self.transfer_predictor:\n            self._make_random_prediction()\n        \n        # Quick predictions for all station pairs\n        if st.button(\"🌐 City-wide Analysis\") and self.transfer_predictor:\n            self._perform_citywide_analysis()\n    \n    def _make_random_prediction(self):\n        \"\"\"Make a random prediction for demonstration\"\"\"\n        target_city = st.session_state.get('target_city')\n        if not target_city:\n            return\n        \n        stations = list(self.city_db.get_city_stations(target_city).keys())\n        if len(stations) < 2:\n            return\n        \n        source_station = np.random.choice(stations)\n        dest_station = np.random.choice([s for s in stations if s != source_station])\n        \n        st.session_state.selected_stations = [source_station, dest_station]\n        self._predict_flow_for_selection()\n        st.rerun()\n    \n    def _perform_citywide_analysis(self):\n        \"\"\"Perform city-wide flow analysis\"\"\"\n        hour = st.session_state.get('selected_hour', 17)\n        day_of_week = st.session_state.get('selected_day_of_week', 1)\n        \n        with st.spinner(\"Analyzing city-wide flow patterns...\"):\n            predictions = self.transfer_predictor.get_all_target_predictions(\n                hour=hour, day_of_week=day_of_week, top_k=10\n            )\n        \n        st.session_state.citywide_predictions = predictions\n        st.success(f\"✅ Analyzed {len(predictions)} flow predictions\")\n    \n    def _show_detailed_analysis(self, prediction: Dict):\n        \"\"\"Show detailed analysis for a prediction\"\"\"\n        st.markdown(\"### 📈 Detailed Flow Analysis\")\n        \n        # Create metrics display\n        col1, col2, col3 = st.columns(3)\n        \n        with col1:\n            st.metric(\n                \"Predicted Flow\",\n                f\"{prediction['flow']:.2f}\",\n                help=\"Number of trips predicted between stations\"\n            )\n        \n        with col2:\n            st.metric(\n                \"Confidence\",\n                f\"{prediction['confidence']:.1%}\",\n                help=\"Model confidence in prediction\"\n            )\n        \n        with col3:\n            flow_category = \"High\" if prediction['flow'] > 5 else \"Medium\" if prediction['flow'] > 2 else \"Low\"\n            st.metric(\n                \"Flow Category\",\n                flow_category,\n                help=\"Categorized flow level\"\n            )\n        \n        # Feature analysis (mock)\n        st.markdown(\"**🗺️ Contributing Factors:**\")\n        st.markdown(\"\"\"\n        - **OSM Features:** High density of restaurants and shops near source station\n        - **Population:** Dense residential area around destination\n        - **Temporal:** Peak commuting hour with high expected mobility\n        - **Transfer Learning:** Pattern similarity with source city data\n        \"\"\")\n    \n    def _render_analysis_section(self):\n        \"\"\"Render bottom analysis section\"\"\"\n        st.markdown(\"---\")\n        \n        # City-wide analysis results\n        if 'citywide_predictions' in st.session_state:\n            predictions = st.session_state.citywide_predictions\n            \n            st.subheader(\"🌐 City-wide Flow Analysis\")\n            \n            if predictions:\n                # Create DataFrame for display\n                df = pd.DataFrame(predictions)\n                \n                # Display top flows\n                st.markdown(\"**Top Predicted Flows:**\")\n                display_df = df[['source', 'target', 'predicted_flow', 'confidence']].copy()\n                display_df.columns = ['Source', 'Target', 'Predicted Flow', 'Confidence']\n                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f\"{x:.1%}\")\n                display_df['Predicted Flow'] = display_df['Predicted Flow'].apply(lambda x: f\"{x:.2f}\")\n                \n                st.dataframe(display_df, use_container_width=True)\n                \n                # Visualization\n                col1, col2 = st.columns(2)\n                \n                with col1:\n                    # Flow distribution\n                    fig_hist = px.histogram(\n                        df, x='predicted_flow',\n                        title=\"Distribution of Predicted Flows\",\n                        labels={'predicted_flow': 'Predicted Flow', 'count': 'Number of Station Pairs'}\n                    )\n                    st.plotly_chart(fig_hist, use_container_width=True)\n                \n                with col2:\n                    # Confidence vs Flow\n                    fig_scatter = px.scatter(\n                        df, x='confidence', y='predicted_flow',\n                        title=\"Prediction Confidence vs Flow\",\n                        labels={'confidence': 'Confidence', 'predicted_flow': 'Predicted Flow'}\n                    )\n                    st.plotly_chart(fig_scatter, use_container_width=True)\n        \n        # Transfer learning insights\n        if self.transfer_predictor:\n            summary = self.transfer_predictor.get_transfer_summary()\n            \n            with st.expander(\"🧠 Transfer Learning Summary\", expanded=False):\n                col1, col2 = st.columns(2)\n                \n                with col1:\n                    st.markdown(f\"\"\"\n                    **Source City:** {summary.get('source_city', 'N/A').title()}\n                    **Target City:** {summary.get('target_city', 'N/A').title()}\n                    **Source Stations:** {summary.get('source_stations', 0)}\n                    **Target Stations:** {summary.get('target_stations', 0)}\n                    \"\"\")\n                \n                with col2:\n                    models = summary.get('models_available', {})\n                    st.markdown(f\"\"\"\n                    **Source Model:** {'✅ Available' if models.get('source_model') else '❌ Not available'}\n                    **Transfer Model:** {'✅ Available' if models.get('transfer_model') else '❌ Not available'}\n                    **Transfer Type:** Domain adaptation enabled\n                    **Features:** OSM + Population + Temporal\n                    \"\"\")\n        \n        # System performance info\n        with st.expander(\"ℹ️ System Information\", expanded=False):\n            st.markdown(\"\"\"\n            ### Cross-City Transfer Learning System\n            \n            **Features:**\n            - 🧠 Graph Neural Network with transfer learning\n            - 🗺️ OpenStreetMap feature extraction\n            - 👥 Population density integration\n            - 🔄 Domain adaptation for cross-city transfer\n            - 📍 Interactive click-to-predict interface\n            \n            **Supported Cities:**\n            - Bern, Zurich, Basel, Geneva, Lausanne (Switzerland)\n            - Extensible to any city with coordinate data\n            \n            **Transfer Learning Process:**\n            1. Train GNN on source city with historical trip data\n            2. Extract OSM and population features for target city\n            3. Align feature distributions between cities\n            4. Apply domain adaptation to reduce city-specific bias\n            5. Generate predictions for target city station pairs\n            \n            **Click-to-Predict Workflow:**\n            1. Select source and target cities\n            2. Initialize transfer learning system\n            3. Click first station (source) on map\n            4. Click second station (destination) on map\n            5. View predicted flow and confidence\n            \"\"\")\n\ndef main():\n    \"\"\"Main application entry point\"\"\"\n    app = InteractiveTransferApp()\n    app.run()\n\nif __name__ == \"__main__\":\n    main()"
