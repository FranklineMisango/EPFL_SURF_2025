"""
Enhanced OpenStreetMap Integration for Interactive Transfer Learning
Provides detailed route visualization and OSM-based path analysis
"""

import folium
from folium.plugins import MarkerCluster, HeatMap, PolyLineTextPath
import numpy as np
import pandas as pd
import requests
import json
import time
from typing import Dict, List, Tuple, Optional, Union
import logging
import math
import streamlit as st
from dataclasses import dataclass

logger = logging.getLogger(__name__)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

@dataclass
class RouteInfo:
    """Information about a route between two stations"""
    source_coords: Tuple[float, float]
    target_coords: Tuple[float, float]
    distance_km: float
    duration_minutes: float
    coordinates: List[Tuple[float, float]]
    route_type: str = "bike"
    
class EnhancedOSMRouter:
    """Enhanced router with OSRM integration and fallback options"""
    
    def __init__(self, osrm_server: str = "http://router.project-osrm.org"):
        self.osrm_server = osrm_server
        
    def get_route_between_stations(
        self, 
        source_coords: Tuple[float, float], 
        target_coords: Tuple[float, float],
        profile: str = "bike"
    ) -> Optional[RouteInfo]:
        """Get detailed route information between two stations"""
        
        try:
            # OSRM API call
            url = f"{self.osrm_server}/route/v1/{profile}/{source_coords[1]},{source_coords[0]};{target_coords[1]},{target_coords[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('routes'):
                    route = data['routes'][0]
                    
                    # Extract route information
                    distance_km = route['distance'] / 1000  # Convert to km
                    duration_min = route['duration'] / 60   # Convert to minutes
                    geometry = route["geometry"]["coordinates"]
                    
                    # Convert coordinates (OSRM returns lon,lat but we need lat,lon)
                    coordinates = [(coord[1], coord[0]) for coord in geometry]
                    
                    route_info = RouteInfo(
                        source_coords=source_coords,
                        target_coords=target_coords,
                        distance_km=distance_km,
                        duration_minutes=duration_min,
                        coordinates=coordinates,
                        route_type=profile
                    )
                    
                    return route_info
                    
        except Exception as e:
            logger.warning(f"Failed to get route from OSRM: {e}")
        
        # Fallback to straight line
        distance = haversine_distance(source_coords[0], source_coords[1], target_coords[0], target_coords[1])
        route_info = RouteInfo(
            source_coords=source_coords,
            target_coords=target_coords,
            distance_km=distance,
            duration_minutes=distance * 4,  # Assume 15 km/h bike speed
            coordinates=[source_coords, target_coords],
            route_type="straight_line"
        )
        
        return route_info

class InteractiveFlowMap:
    """Enhanced map creation with interactive features"""
    
    def __init__(self, osm_router: EnhancedOSMRouter):
        self.osm_router = osm_router
        
    def create_enhanced_map(
        self,
        center_coords: Tuple[float, float],
        stations: Dict[str, Tuple[float, float]],
        selected_stations: List[str] = None,
        predictions: List[Dict] = None,
        show_heatmap: bool = False
    ) -> folium.Map:
        """Create an enhanced interactive map with stations and features"""
        
        # Initialize map
        m = folium.Map(
            location=list(center_coords),
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('CartoDB positron').add_to(m)
        folium.TileLayer('OpenStreetMap').add_to(m)
        
        # Create marker cluster for stations
        marker_cluster = MarkerCluster(name="Bike Stations").add_to(m)
        
        selected_stations = selected_stations or []
        
        # Add station markers
        for station_id, coords in stations.items():
            # Determine marker style based on selection
            if station_id in selected_stations:
                color = 'red' if station_id == selected_stations[0] else 'green'
                icon = 'star'
            else:
                color = 'blue'
                icon = 'bicycle'
            
            # Create popup content
            popup_content = self._create_station_popup(station_id, coords)
            
            folium.Marker(
                location=list(coords),
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Station {station_id}",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(marker_cluster)
        
        # Add route visualization if stations are selected
        if len(selected_stations) >= 2:
            self._add_route_visualization(m, stations, selected_stations)
        
        # Add prediction heatmap if available
        if show_heatmap and predictions:
            self._add_prediction_heatmap(m, predictions, stations)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def _create_station_popup(self, station_id: str, coords: Tuple[float, float]) -> str:
        """Create detailed popup content for a station"""
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 250px;">
            <h4 style="color: #1f77b4; margin-bottom: 10px;">🚲 Station {station_id}</h4>
            <p><strong>📍 Location:</strong><br>
            {coords[0]:.4f}, {coords[1]:.4f}</p>
            <p><strong>Click to select for prediction</strong></p>
        </div>
        """
        return popup_html
    
    def _add_route_visualization(
        self, 
        m: folium.Map, 
        stations: Dict[str, Tuple[float, float]], 
        selected_stations: List[str]
    ):
        """Add route visualization between selected stations"""
        if len(selected_stations) < 2:
            return
        
        source_coords = stations[selected_stations[0]]
        target_coords = stations[selected_stations[1]]
        
        # Get route information
        route_info = self.osm_router.get_route_between_stations(source_coords, target_coords)
        
        if route_info:
            # Add route line
            route_color = 'red' if route_info.route_type == 'straight_line' else 'blue'
            
            folium.PolyLine(
                locations=route_info.coordinates,
                color=route_color,
                weight=4,
                opacity=0.8,
                popup=f"Distance: {route_info.distance_km:.2f} km, Duration: {route_info.duration_minutes:.1f} min"
            ).add_to(m)
            
            # Add route markers
            folium.Marker(
                location=route_info.source_coords,
                popup=f"Source: {selected_stations[0]}",
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(m)
            
            folium.Marker(
                location=route_info.target_coords,
                popup=f"Destination: {selected_stations[1]}",
                icon=folium.Icon(color='red', icon='stop', prefix='fa')
            ).add_to(m)
    
    def _add_prediction_heatmap(
        self, 
        m: folium.Map, 
        predictions: List[Dict], 
        stations: Dict[str, Tuple[float, float]]
    ):
        """Add heatmap overlay based on predictions"""
        heat_data = []
        
        for pred in predictions:
            source_coords = stations.get(pred['source'])
            target_coords = stations.get(pred['target'])
            
            if source_coords and target_coords:
                # Add heat points based on predicted flow
                intensity = min(pred['predicted_flow'] / 10.0, 1.0)  # Normalize
                
                # Add points along the route
                heat_data.append([source_coords[0], source_coords[1], intensity])
                heat_data.append([target_coords[0], target_coords[1], intensity])
        
        if heat_data:
            HeatMap(heat_data, name="Flow Intensity").add_to(m)

def create_enhanced_transfer_map(
    center_coords: Tuple[float, float],
    stations: Dict[str, Tuple[float, float]],
    selected_stations: List[str] = None,
    predictions: List[Dict] = None,
    show_heatmap: bool = False
) -> folium.Map:
    """
    Create an enhanced transfer learning map with interactive features
    
    Args:
        center_coords: Center coordinates for the map
        stations: Dictionary of station_id -> (lat, lon)
        selected_stations: List of selected station IDs
        predictions: List of prediction dictionaries
        show_heatmap: Whether to show prediction heatmap
        
    Returns:
        Folium map object
    """
    
    # Initialize router and map creator
    osm_router = EnhancedOSMRouter()
    map_creator = InteractiveFlowMap(osm_router)
    
    # Create the enhanced map
    enhanced_map = map_creator.create_enhanced_map(
        center_coords=center_coords,
        stations=stations,
        selected_stations=selected_stations,
        predictions=predictions,
        show_heatmap=show_heatmap
    )
    
    return enhanced_map

def create_route_analysis_map(
    source_coords: Tuple[float, float],
    target_coords: Tuple[float, float],
    route_alternatives: List[RouteInfo] = None
) -> folium.Map:
    """
    Create a detailed route analysis map
    
    Args:
        source_coords: Source station coordinates
        target_coords: Target station coordinates
        route_alternatives: List of alternative routes
        
    Returns:
        Folium map object with route analysis
    """
    
    # Calculate center point
    center_lat = (source_coords[0] + target_coords[0]) / 2
    center_lon = (source_coords[1] + target_coords[1]) / 2
    
    # Initialize map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add source and destination markers
    folium.Marker(
        location=source_coords,
        popup="Source Station",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        location=target_coords,
        popup="Destination Station",
        icon=folium.Icon(color='red', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Add route alternatives if provided
    if route_alternatives:
        colors = ['blue', 'purple', 'orange', 'darkred']
        
        for i, route in enumerate(route_alternatives[:4]):  # Max 4 alternatives
            color = colors[i % len(colors)]
            
            folium.PolyLine(
                locations=route.coordinates,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Route {i+1}: {route.distance_km:.2f} km, {route.duration_minutes:.1f} min"
            ).add_to(m)
    
    return m

def analyze_osm_features_for_route(
    source_coords: Tuple[float, float],
    target_coords: Tuple[float, float],
    buffer_meters: float = 500
) -> Dict:
    """
    Analyze OSM features along a route
    
    Args:
        source_coords: Source coordinates
        target_coords: Target coordinates
        buffer_meters: Buffer distance for feature extraction
        
    Returns:
        Dictionary with route analysis
    """
    
    # Initialize router
    router = EnhancedOSMRouter()
    
    # Get route information
    route_info = router.get_route_between_stations(source_coords, target_coords)
    
    if not route_info:
        return {"error": "Could not analyze route"}
    
    # Analyze route characteristics
    analysis = {
        "route_info": {
            "distance_km": route_info.distance_km,
            "duration_minutes": route_info.duration_minutes,
            "route_type": route_info.route_type,
            "coordinates_count": len(route_info.coordinates)
        },
        "efficiency": {
            "straight_line_distance": haversine_distance(
                source_coords[0], source_coords[1], 
                target_coords[0], target_coords[1]
            ),
            "route_efficiency": 0.0,
            "speed_kmh": 0.0
        }
    }
    
    # Calculate efficiency metrics
    straight_distance = analysis["efficiency"]["straight_line_distance"]
    if straight_distance > 0:
        analysis["efficiency"]["route_efficiency"] = straight_distance / route_info.distance_km
    
    if route_info.duration_minutes > 0:
        analysis["efficiency"]["speed_kmh"] = (route_info.distance_km / route_info.duration_minutes) * 60
    
    return analysis

def create_flow_prediction_visualization(
    predictions: List[Dict],
    stations: Dict[str, Tuple[float, float]],
    center_coords: Tuple[float, float]
) -> folium.Map:
    """
    Create a visualization of flow predictions across the city
    
    Args:
        predictions: List of prediction dictionaries
        stations: Station coordinates dictionary
        center_coords: Map center coordinates
        
    Returns:
        Folium map with flow visualizations
    """
    
    # Initialize map
    m = folium.Map(
        location=list(center_coords),
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Process predictions for visualization
    for pred in predictions:
        source_coords = stations.get(pred['source'])
        target_coords = stations.get(pred['target'])
        
        if source_coords and target_coords:
            # Determine line style based on flow magnitude
            flow = pred['predicted_flow']
            confidence = pred['confidence']
            
            # Color based on flow magnitude
            if flow > 5:
                color = 'red'
                weight = 5
            elif flow > 2:
                color = 'orange' 
                weight = 3
            else:
                color = 'blue'
                weight = 2
            
            # Opacity based on confidence
            opacity = max(0.3, confidence)
            
            # Add flow line
            folium.PolyLine(
                locations=[source_coords, target_coords],
                color=color,
                weight=weight,
                opacity=opacity,
                popup=f"Flow: {flow:.2f}, Confidence: {confidence:.1%}"
            ).add_to(m)
    
    # Add stations
    for station_id, coords in stations.items():
        folium.CircleMarker(
            location=coords,
            radius=4,
            popup=f"Station {station_id}",
            color='black',
            fillColor='white',
            fillOpacity=0.8
        ).add_to(m)
    
    return m
