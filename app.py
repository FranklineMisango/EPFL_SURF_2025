import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import networkx as nx
import pickle
import ast
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
import os
import time
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bike Flow Prediction System",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_bike_data():
    """Load and preprocess bike trip data"""
    try:
        trips_df = pd.read_csv('data/trips_8days_flat.csv')
        
        # Parse timestamps
        trips_df['start_datetime'] = pd.to_datetime(trips_df['start_time'], format='%Y%m%d_%H%M%S')
        trips_df['end_datetime'] = pd.to_datetime(trips_df['end_time'], format='%Y%m%d_%H%M%S')
        trips_df['hour'] = trips_df['start_datetime'].dt.hour
        
        # Extract coordinates
        trips_df['start_lat'] = trips_df['start_coords'].apply(lambda x: ast.literal_eval(x)[0])
        trips_df['start_lon'] = trips_df['start_coords'].apply(lambda x: ast.literal_eval(x)[1])
        trips_df['end_lat'] = trips_df['end_coords'].apply(lambda x: ast.literal_eval(x)[0])
        trips_df['end_lon'] = trips_df['end_coords'].apply(lambda x: ast.literal_eval(x)[1])
        
        return trips_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def create_simple_routing_network(trips_df):
    """Create a simplified routing network from actual trip data"""
    try:
        # Create routing lookup from actual trips
        routing_network = {}
        
        # Group trips to find common routes
        route_counts = trips_df.groupby(['start_station_id', 'end_station_id']).agg({
            'start_lat': 'first',
            'start_lon': 'first', 
            'end_lat': 'first',
            'end_lon': 'first'
        }).reset_index()
        
        for _, row in route_counts.iterrows():
            start_station = row['start_station_id']
            end_station = row['end_station_id']
            
            if start_station != end_station:  # Skip self-loops
                # Create simple route (can be enhanced with actual path finding)
                start_coords = [row['start_lat'], row['start_lon']]
                end_coords = [row['end_lat'], row['end_lon']]
                
                # For now, create a simple 2-point route (can be enhanced)
                route_key = f"{start_station}_{end_station}"
                routing_network[route_key] = [start_coords, end_coords]
        
        return routing_network
    except Exception as e:
        st.warning(f"Could not create routing network: {e}")
        return {}

class FastBikeFlowPredictor:
    """Optimized bike flow predictor with fast loading"""
    
    def __init__(self, trips_df, routing_network=None):
        self.trips_df = trips_df
        self.routing_network = routing_network or {}
        self.station_coords = {}
        self.hourly_flows = {}
        self.model = None
        self.scaler = None
        self.station_features = {}
        
        # Initialize with progress tracking
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        """Initialize predictor with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Building station network...")
        self._build_station_network()
        progress_bar.progress(25)
        
        status_text.text("Computing hourly flows...")
        self._compute_hourly_flows()
        progress_bar.progress(50)
        
        status_text.text("Engineering features...")
        self._engineer_features()
        progress_bar.progress(75)
        
        status_text.text("Training ML model...")
        self._train_model()
        progress_bar.progress(100)
        
        status_text.text("✅ Ready!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    def _build_station_network(self):
        """Build station coordinate mapping"""
        # Get unique stations
        start_stations = self.trips_df.groupby('start_station_id').agg({
            'start_lat': 'first',
            'start_lon': 'first'
        }).reset_index()
        
        for _, row in start_stations.iterrows():
            station_id = row['start_station_id']
            self.station_coords[station_id] = (row['start_lat'], row['start_lon'])
    
    def _compute_hourly_flows(self):
        """Compute flow volumes by hour"""
        for hour in range(24):
            hour_trips = self.trips_df[self.trips_df['hour'] == hour]
            flows = hour_trips.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='flow_count')
            self.hourly_flows[hour] = flows
    
    def _engineer_features(self):
        """Engineer features for ML model"""
        for station_id in self.station_coords.keys():
            features = {}
            
            station_trips = self.trips_df[
                (self.trips_df['start_station_id'] == station_id) | 
                (self.trips_df['end_station_id'] == station_id)
            ]
            
            features['total_trips'] = len(station_trips)
            features['avg_temperature'] = station_trips['temperature'].mean() if len(station_trips) > 0 else 20.0
            features['lat'] = self.station_coords[station_id][0]
            features['lon'] = self.station_coords[station_id][1]
            
            # Hourly patterns
            for hour in range(24):
                outflow = len(station_trips[
                    (station_trips['start_station_id'] == station_id) & 
                    (station_trips['hour'] == hour)
                ])
                inflow = len(station_trips[
                    (station_trips['end_station_id'] == station_id) & 
                    (station_trips['hour'] == hour)
                ])
                
                features[f'outflow_hour_{hour}'] = outflow
                features[f'inflow_hour_{hour}'] = inflow
            
            self.station_features[station_id] = features
    
    def _train_model(self):
        """Train ML model for flow prediction"""
        training_data = []
        
        # Sample data for faster training
        sample_hours = [6, 8, 12, 17, 20]  # Focus on key hours
        
        for hour in sample_hours:
            if hour in self.hourly_flows:
                flows = self.hourly_flows[hour]
                
                # Sample flows for faster training
                if len(flows) > 1000:
                    flows = flows.sample(n=1000, random_state=42)
                
                for _, row in flows.iterrows():
                    start_station = row['start_station_id']
                    end_station = row['end_station_id']
                    flow_count = row['flow_count']
                    
                    if start_station in self.station_features and end_station in self.station_features:
                        feature_vector = self._create_feature_vector(start_station, end_station, hour)
                        training_data.append(feature_vector + [flow_count])
        
        if training_data:
            training_data = np.array(training_data)
            X = training_data[:, :-1]
            y = training_data[:, -1]
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Use fewer estimators for faster training
            self.model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            self.model.fit(X_scaled, y)
    
    def _create_feature_vector(self, start_station, end_station, hour):
        """Create feature vector for prediction"""
        start_features = self.station_features[start_station]
        end_features = self.station_features[end_station]
        
        feature_vector = [
            hour,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            start_features['lat'],
            start_features['lon'],
            start_features['total_trips'],
            start_features[f'outflow_hour_{hour}'],
            end_features['lat'],
            end_features['lon'],
            end_features['total_trips'],
            end_features[f'inflow_hour_{hour}'],
            np.sqrt((start_features['lat'] - end_features['lat'])**2 + 
                   (start_features['lon'] - end_features['lon'])**2),
            (start_features['avg_temperature'] + end_features['avg_temperature']) / 2
        ]
        
        return feature_vector
    
    def predict_destinations(self, station_id, hour, top_k=5):
        """Predict top destinations from a station with confidence levels"""
        if station_id not in self.station_features or not self.model:
            return []
        
        predictions = []
        
        # Only check active destinations for this hour to speed up
        active_destinations = set()
        if hour in self.hourly_flows:
            hour_flows = self.hourly_flows[hour]
            station_flows = hour_flows[hour_flows['start_station_id'] == station_id]
            active_destinations = set(station_flows['end_station_id'].values)
        
        # If no active destinations, use top destinations from nearby hours
        if not active_destinations:
            nearby_hours = [(hour-1) % 24, hour, (hour+1) % 24]
            for h in nearby_hours:
                if h in self.hourly_flows:
                    hour_flows = self.hourly_flows[h]
                    station_flows = hour_flows[hour_flows['start_station_id'] == station_id]
                    active_destinations.update(station_flows['end_station_id'].values)
                if len(active_destinations) >= 20:  # Limit for performance
                    break
        
        # Predict for active destinations only
        for dest_station in active_destinations:
            if dest_station == station_id or dest_station not in self.station_features:
                continue
            
            feature_vector = self._create_feature_vector(station_id, dest_station, hour)
            X_pred = self.scaler.transform([feature_vector])
            predicted_flow = self.model.predict(X_pred)[0]
            
            # Calculate confidence based on prediction and historical data
            confidence = min(0.95, max(0.1, predicted_flow / 10))
            
            predictions.append({
                'destination': dest_station,
                'predicted_flow': max(0, predicted_flow),
                'confidence': confidence
            })
        
        # Sort by predicted flow and return top k
        predictions.sort(key=lambda x: x['predicted_flow'], reverse=True)
        return predictions[:top_k]
    
    def get_route_path(self, start_station, end_station):
        """Get route path between stations"""
        route_key = f"{start_station}_{end_station}"
        
        if route_key in self.routing_network:
            return self.routing_network[route_key]
        else:
            # Fallback to straight line
            if start_station in self.station_coords and end_station in self.station_coords:
                start_lat, start_lon = self.station_coords[start_station]
                end_lat, end_lon = self.station_coords[end_station]
                return [[start_lat, start_lon], [end_lat, end_lon]]
            else:
                return []

def create_interactive_map(predictor, selected_hour=17, selected_station=None):
    """Create interactive satellite map with predictions"""
    
    # Get center coordinates
    all_coords = list(predictor.station_coords.values())
    center_lat = np.mean([coord[0] for coord in all_coords])
    center_lon = np.mean([coord[1] for coord in all_coords])
    
    # Create map with satellite imagery
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None
    )
    
    # Add satellite tiles
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add OpenStreetMap overlay for reference
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Sample stations for performance (show top 30 most active)
    station_activity = {}
    for station_id in predictor.station_coords.keys():
        activity = predictor.station_features[station_id]['total_trips']
        station_activity[station_id] = activity
    
    top_stations = sorted(station_activity.items(), key=lambda x: x[1], reverse=True)[:30]
    
    # Add station markers
    for station_id, _ in top_stations:
        lat, lon = predictor.station_coords[station_id]
        
        # Determine marker color based on selection
        if station_id == selected_station:
            color = 'red'
            icon = 'star'
        else:
            color = 'blue'
            icon = 'bicycle'
        
        # Create popup with basic info
        popup_text = f"""
        <b>Station {station_id}</b><br>
        Total Trips: {predictor.station_features[station_id]['total_trips']}<br>
        Click to see predictions!
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Station {station_id}",
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m)
    
    # Add prediction routes if station is selected
    if selected_station and selected_station in predictor.station_coords:
        predictions = predictor.predict_destinations(selected_station, selected_hour, top_k=5)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, pred in enumerate(predictions):
            dest_station = pred['destination']
            flow = pred['predicted_flow']
            confidence = pred['confidence']
            
            if dest_station in predictor.station_coords and flow > 0.5:
                # Get route path
                route_coords = predictor.get_route_path(selected_station, dest_station)
                
                if route_coords:
                    # Calculate line properties
                    weight = max(2, min(8, flow))
                    opacity = max(0.3, confidence)
                    color = colors[i % len(colors)]
                    
                    # Add route line
                    folium.PolyLine(
                        locations=route_coords,
                        color=color,
                        weight=weight,
                        opacity=opacity,
                        popup=f"""
                        <b>Route to Station {dest_station}</b><br>
                        Predicted Flow: {flow:.1f} trips<br>
                        Confidence: {confidence:.2%}<br>
                        Rank: #{i+1}
                        """
                    ).add_to(m)
                    
                    # Add destination marker
                    dest_lat, dest_lon = predictor.station_coords[dest_station]
                    folium.CircleMarker(
                        location=[dest_lat, dest_lon],
                        radius=max(5, flow),
                        popup=f"Destination {dest_station}<br>Flow: {flow:.1f}",
                        color=color,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🚴‍♂️ Bike Flow Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("*⚡ Optimized for fast loading with intelligent caching*")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Load data
    with st.spinner("Loading bike data..."):
        trips_df = load_bike_data()
        if trips_df is None:
            st.error("Failed to load bike data. Please check the data file.")
            return
    
    # Create routing network
    routing_network = create_simple_routing_network(trips_df)
    
    # Initialize predictor
    @st.cache_resource
    def get_predictor():
        return FastBikeFlowPredictor(trips_df, routing_network)
    
    st.markdown("### 🔄 Initializing Prediction System...")
    predictor = get_predictor()
    
    # Time slider
    selected_hour = st.sidebar.slider(
        "🕐 Select Hour",
        min_value=0,
        max_value=23,
        value=17,
        help="Select the hour of day for predictions"
    )
    
    # Station selection
    available_stations = sorted(list(predictor.station_coords.keys()))
    selected_station = st.sidebar.selectbox(
        "🚉 Select Station (Optional)",
        options=[None] + available_stations[:20],  # Limit options for performance
        help="Select a station to see its predicted destinations"
    )
    
    # Performance info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Performance")
    st.sidebar.info(f"""
    **Stations**: {len(predictor.station_coords)}
    **Active Routes**: {len(routing_network)}
    **Model**: Fast Random Forest
    **Cache**: 24h data retention
    """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Interactive Satellite Map")
        st.info("💡 Click on any blue station marker to see flow predictions!")
        
        # Create and display map
        map_obj = create_interactive_map(predictor, selected_hour, selected_station)
        map_data = st_folium(map_obj, width=800, height=600, returned_objects=["last_object_clicked"])
        
        # Handle map clicks
        if map_data['last_object_clicked']:
            clicked_data = map_data['last_object_clicked']
            if 'tooltip' in clicked_data:
                tooltip = clicked_data['tooltip']
                if 'Station' in tooltip:
                    try:
                        clicked_station_id = float(tooltip.split('Station ')[1])
                        if clicked_station_id in predictor.station_coords:
                            st.session_state.selected_station = clicked_station_id
                            st.rerun()
                    except:
                        pass
    
    with col2:
        st.subheader("📊 Prediction Details")
        
        if selected_station:
            # Show station info
            st.markdown(f"""
            <div class="metric-card">
                <h4>🚉 Station {selected_station}</h4>
                <p><strong>Time:</strong> {selected_hour:02d}:00</p>
                <p><strong>Total Historical Trips:</strong> {predictor.station_features[selected_station]['total_trips']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get and display predictions
            with st.spinner("Calculating predictions..."):
                predictions = predictor.predict_destinations(selected_station, selected_hour, top_k=8)
            
            if predictions:
                st.subheader("🎯 Top Predicted Destinations")
                
                for i, pred in enumerate(predictions[:5]):
                    dest = pred['destination']
                    flow = pred['predicted_flow']
                    confidence = pred['confidence']
                    
                    if flow > 0.1:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h5>#{i+1} Station {dest}</h5>
                            <p><strong>Predicted Flow:</strong> {flow:.1f} trips</p>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Create confidence chart
                if len(predictions) > 0:
                    fig = go.Figure()
                    
                    destinations = [f"Station {p['destination']}" for p in predictions[:5]]
                    flows = [p['predicted_flow'] for p in predictions[:5]]
                    
                    fig.add_trace(go.Bar(
                        x=destinations,
                        y=flows,
                        name='Predicted Flow',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title="Predicted Flows by Destination",
                        xaxis_title="Destination",
                        yaxis_title="Predicted Flow (trips)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant predictions for this station at this time.")
        else:
            st.info("👆 Click on a station on the map to see predictions!")
            
            # Show overall statistics
            st.subheader("📈 System Overview")
            
            total_stations = len(predictor.station_coords)
            total_trips = len(trips_df)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Stations", total_stations)
            with col_b:
                st.metric("Total Trips", f"{total_trips:,}")
            
            # Hourly activity chart
            hourly_activity = trips_df.groupby('hour').size()
            
            fig = px.line(
                x=hourly_activity.index,
                y=hourly_activity.values,
                title="Bike Activity Throughout the Day",
                labels={'x': 'Hour', 'y': 'Number of Trips'}
            )
            fig.add_vline(x=selected_hour, line_dash="dash", line_color="red", 
                         annotation_text=f"Selected: {selected_hour}:00")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 How to Use:
    1. **Select Time**: Use the hour slider in the sidebar
    2. **Click Station**: Click on any blue bike station marker
    3. **View Predictions**: See predicted destinations with confidence levels
    4. **Explore**: Try different times and stations!
    
    ### ⚡ Performance Optimizations:
    - **Smart Caching**: 24-hour data retention for fast reloading
    - **Focused Training**: ML model trained on key hours only
    - **Limited Stations**: Top 30 most active stations shown
    - **Efficient Routing**: Simplified path calculation
    - **Progressive Loading**: Step-by-step initialization with progress tracking
    """)

if __name__ == "__main__":
    main()
