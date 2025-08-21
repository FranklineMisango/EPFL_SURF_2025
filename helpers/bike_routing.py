"""
Bike routing system using OSMnx to get actual bike paths and roads
"""
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
import pickle
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

class BikeRouter:
    """Handle bike routing using actual road networks"""
    
    def __init__(self, cache_dir="cache/routing"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.graphs = {}  # Cache for city graphs
        
    @lru_cache(maxsize=10)
    def get_city_graph(self, city_name: str, bbox: Tuple[float, float, float, float]) -> Optional[nx.MultiDiGraph]:
        """Get or create bike network graph for a city"""
        cache_file = os.path.join(self.cache_dir, f"{city_name}_bike_graph.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached graph for {city_name}: {e}")
        
        try:
            # Create bounding box (north, south, east, west)
            north, south, east, west = bbox
            
            # Download bike network
            logger.info(f"Downloading bike network for {city_name}...")
            G = ox.graph_from_bbox(
                north, south, east, west,
                network_type='bike',  # Focus on bike-friendly roads
                simplify=True,
                retain_all=False
            )
            
            # Add edge speeds and travel times
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            
            # Cache the graph
            with open(cache_file, 'wb') as f:
                pickle.dump(G, f)
            
            logger.info(f"Successfully created bike network for {city_name}")
            return G
            
        except Exception as e:
            logger.error(f"Failed to create bike network for {city_name}: {e}")
            return None
    
    def get_route_path(self, start_coords: Tuple[float, float], 
                      end_coords: Tuple[float, float], 
                      city_graph: nx.MultiDiGraph) -> List[Tuple[float, float]]:
        """Get actual bike route between two points"""
        try:
            start_lat, start_lon = start_coords
            end_lat, end_lon = end_coords
            
            # Find nearest nodes
            start_node = ox.nearest_nodes(city_graph, start_lon, start_lat)
            end_node = ox.nearest_nodes(city_graph, end_lon, end_lat)
            
            # Calculate shortest path
            route = nx.shortest_path(city_graph, start_node, end_node, weight='travel_time')
            
            # Convert to coordinates
            route_coords = []
            for node in route:
                node_data = city_graph.nodes[node]
                route_coords.append((node_data['y'], node_data['x']))  # (lat, lon)
            
            return route_coords
            
        except Exception as e:
            logger.warning(f"Failed to find route: {e}")
            # Fallback to straight line
            return [start_coords, end_coords]
    
    def get_multiple_routes(self, source_coords: Tuple[float, float], 
                           destinations: List[Tuple[str, Tuple[float, float]]], 
                           city_name: str, bbox: Tuple[float, float, float, float]) -> Dict[str, List[Tuple[float, float]]]:
        """Get routes from source to multiple destinations"""
        routes = {}
        
        # Get city graph
        city_graph = self.get_city_graph(city_name, bbox)
        if not city_graph:
            logger.warning(f"No graph available for {city_name}, using straight lines")
            for dest_id, dest_coords in destinations:
                routes[dest_id] = [source_coords, dest_coords]
            return routes
        
        # Calculate routes
        for dest_id, dest_coords in destinations:
            route_path = self.get_route_path(source_coords, dest_coords, city_graph)
            routes[dest_id] = route_path
        
        return routes

# City bounding boxes for major Swiss cities
CITY_BBOXES = {
    'Zurich': (47.46, 47.32, 8.67, 8.46),      # (north, south, east, west)
    'Bern': (46.99, 46.90, 7.52, 7.35),
    'Geneva': (46.48, 46.15, 6.36, 6.12),
    'Basel': (47.60, 47.52, 7.65, 7.57),
    'Lausanne': (46.56, 46.49, 6.68, 6.55),
    'Lugano': (46.07, 45.80, 9.06, 8.87),
    'Fribourg': (46.84, 46.76, 7.19, 7.09),
    'St. Gallen': (47.46, 47.39, 9.56, 9.49),
}

def get_city_bbox(city_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Get bounding box for a city"""
    return CITY_BBOXES.get(city_name)

# Global router instance
_router = None

def get_bike_router() -> BikeRouter:
    """Get global bike router instance"""
    global _router
    if _router is None:
        _router = BikeRouter()
    return _router