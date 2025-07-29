"""
Optimized map creation for better performance
"""

import folium
from folium.plugins import MarkerCluster
import numpy as np
import streamlit as st

def create_optimized_map(predictor, selected_hour=17, selected_day_of_week=1, selected_station=None):
    """Create optimized map with better performance for zooming"""
    
    # Get center coordinates
    all_coords = list(predictor.station_coords.values())
    center_lat = np.mean([coord[0] for coord in all_coords])
    center_lon = np.mean([coord[1] for coord in all_coords])
    
    # Create map with performance optimizations
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap',  # Use single default tile layer
        prefer_canvas=True,     # Use canvas for better performance
        max_zoom=18,
        min_zoom=10
    )
    
    # Create marker cluster for better performance
    marker_cluster = MarkerCluster(
        name="Bike Stations",
        overlay=True,
        control=True,
        options={
            'maxClusterRadius': 50,
            'disableClusteringAtZoom': 15,
            'spiderfyOnMaxZoom': True,
            'chunkedLoading': True  # Load markers in chunks
        }
    ).add_to(m)
    
    # Limit stations for performance
    if selected_station:
        # Show all stations when one is selected
        stations_to_show = list(predictor.station_coords.keys())
    else:
        # Show only subset for better initial performance
        stations_to_show = list(predictor.station_coords.keys())[:30]
    
    # Add station markers with simplified content
    for station_id in stations_to_show:
        if station_id not in predictor.station_coords:
            continue
            
        lat, lon = predictor.station_coords[station_id]
        
        # Simplified popup for better performance
        total_trips = predictor.station_features.get(station_id, {}).get('total_trips', 0)
        popup_content = f"<b>Station {station_id}</b><br>Trips: {total_trips}"
        
        # Add detailed OSM features only for selected station
        if station_id == selected_station:
            osm_features = predictor.osm_features.get(station_id, {})
            if osm_features:
                popup_content += "<br><b>Features:</b><br>"
                feature_count = 0
                for feature, count in osm_features.items():
                    if 'count' in feature and count > 0 and feature_count < 3:
                        feature_name = feature.replace('_count', '').replace('_', ' ').title()
                        popup_content += f"{feature_name}: {count}<br>"
                        feature_count += 1
        
        # Different handling for selected vs regular stations
        if station_id == selected_station:
            # Selected station - prominent marker, not clustered
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=250),
                tooltip=f"Selected: {station_id}",
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(m)
        else:
            # Regular stations - add to cluster with simple markers
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=200),
                tooltip=f"Station {station_id}",
                icon=folium.Icon(color='blue', icon='bicycle', prefix='fa')
            ).add_to(marker_cluster)
    
    # Add predictions with optimized paths
    if selected_station and selected_station in predictor.station_coords:
        # Create feature group for paths
        path_group = folium.FeatureGroup(
            name="Predicted Routes", 
            overlay=True, 
            control=True
        )
        
        # Get predictions (reduced number for performance)
        predictions = predictor.predict_with_multiple_paths(
            selected_station, selected_hour, 
            day_of_week=selected_day_of_week, top_k=3
        )
        
        colors = ['#FF0000', '#0000FF', '#00FF00']  # Hex colors for better performance
        
        for i, pred in enumerate(predictions):
            dest_station = pred['destination']
            flow = pred['predicted_flow']
            confidence = pred['confidence']
            paths = pred['paths']
            
            if dest_station in predictor.station_coords and paths:
                # Use only the best path for performance
                best_path = paths[0]
                
                if best_path.coordinates and len(best_path.coordinates) > 1:
                    # Simplify coordinates for better performance
                    coords = best_path.coordinates
                    if len(coords) > 20:
                        # Take every nth point to reduce complexity
                        step = len(coords) // 20
                        coords = coords[::step]
                    
                    # Simplified popup
                    popup_content = f"""
                    <b>To Station {dest_station}</b><br>
                    Type: {best_path.path_type.title()}<br>
                    Distance: {best_path.distance_m/1000:.1f}km<br>
                    Flow: {flow:.1f} trips
                    """
                    
                    folium.PolyLine(
                        locations=coords,
                        color=colors[i % len(colors)],
                        weight=4,
                        opacity=0.8,
                        popup=folium.Popup(popup_content, max_width=200),
                        tooltip=f"To {dest_station}"
                    ).add_to(path_group)
                
                # Add destination marker
                dest_lat, dest_lon = predictor.station_coords[dest_station]
                folium.CircleMarker(
                    location=[dest_lat, dest_lon],
                    radius=min(15, max(8, flow)),
                    popup=f"Dest: {dest_station}<br>Flow: {flow:.1f}",
                    color=colors[i % len(colors)],
                    fillColor=colors[i % len(colors)],
                    fillOpacity=0.6,
                    weight=2
                ).add_to(path_group)
        
        path_group.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m