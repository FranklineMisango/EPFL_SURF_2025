"""
Multi-Path Router for Bike Flow Prediction
Generates multiple possible paths between stations, not just the shortest route.
Considers various routing preferences and path alternatives.
"""

import requests
import json
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from geopy.distance import geodesic
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PathInfo:
    """Information about a specific path between two points"""
    path_id: str
    coordinates: List[Tuple[float, float]]  # [(lat, lon), ...]
    distance_m: float
    duration_s: float
    path_type: str  # 'shortest', 'fastest', 'safest', 'scenic', 'alternative'
    routing_profile: str  # 'cycling-regular', 'cycling-safe', 'cycling-mountain', etc.
    elevation_gain: float = 0.0
    surface_types: List[str] = None
    traffic_level: str = 'unknown'  # 'low', 'medium', 'high'
    bike_infrastructure: float = 0.0  # 0-1 score for bike lanes/paths
    confidence: float = 1.0
    source: str = 'unknown'

class MultiPathRouter:
    """Generate multiple routing options between bike stations"""
    
    def __init__(self, cache_dir: str = "cache/multi_paths"):
        self.cache_dir = cache_dir
        self.routing_profiles = self._define_routing_profiles()
        self.path_types = self._define_path_types()
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        import os
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _define_routing_profiles(self) -> Dict[str, Dict]:
        """Define different cycling routing profiles"""
        return {
            'cycling-regular': {
                'description': 'Standard cycling route',
                'preference': 'balanced',
                'avoid': [],
                'priority': 'time'
            },
            'cycling-safe': {
                'description': 'Safest cycling route with bike infrastructure',
                'preference': 'safety',
                'avoid': ['highways', 'busy_roads'],
                'priority': 'safety'
            },
            'cycling-fast': {
                'description': 'Fastest cycling route',
                'preference': 'speed',
                'avoid': [],
                'priority': 'time'
            },
            'cycling-scenic': {
                'description': 'Scenic route through parks and quiet areas',
                'preference': 'scenery',
                'avoid': ['highways', 'industrial'],
                'priority': 'experience'
            },
            'cycling-direct': {
                'description': 'Most direct route',
                'preference': 'distance',
                'avoid': [],
                'priority': 'distance'
            }
        }
    
    def _define_path_types(self) -> Dict[str, Dict]:
        """Define different types of paths to generate"""
        return {
            'shortest': {
                'description': 'Shortest distance path',
                'weight': 'distance',
                'alternatives': False
            },
            'fastest': {
                'description': 'Fastest time path',
                'weight': 'time',
                'alternatives': False
            },
            'safest': {
                'description': 'Safest path with bike infrastructure',
                'weight': 'safety',
                'alternatives': False
            },
            'alternative_1': {
                'description': 'First alternative route',
                'weight': 'distance',
                'alternatives': True
            },
            'alternative_2': {
                'description': 'Second alternative route',
                'weight': 'time',
                'alternatives': True
            },
            'scenic': {
                'description': 'Scenic route through parks',
                'weight': 'scenery',
                'alternatives': False
            }
        }
    
    def get_multiple_paths(self, start_lat: float, start_lon: float, 
                          end_lat: float, end_lon: float,
                          max_paths: int = 5) -> List[PathInfo]:
        """Get multiple path options between two points"""
        
        # Generate cache key
        cache_key = f"multi_{start_lat:.6f}_{start_lon:.6f}_{end_lat:.6f}_{end_lon:.6f}_{max_paths}"
        
        # Try to load from cache
        cached_paths = self._load_paths_from_cache(cache_key)
        if cached_paths:
            return cached_paths[:max_paths]
        
        paths = []
        
        # 1. Try OpenRouteService for multiple routes
        ors_paths = self._get_openrouteservice_alternatives(start_lat, start_lon, end_lat, end_lon)
        paths.extend(ors_paths)
        
        # 2. Try OSRM for alternatives
        if len(paths) < max_paths:
            osrm_paths = self._get_osrm_alternatives(start_lat, start_lon, end_lat, end_lon)
            paths.extend(osrm_paths)
        
        # 3. Generate synthetic alternatives if needed
        if len(paths) < max_paths:
            synthetic_paths = self._generate_synthetic_alternatives(
                start_lat, start_lon, end_lat, end_lon, 
                existing_paths=paths,
                needed_count=max_paths - len(paths)
            )
            paths.extend(synthetic_paths)
        
        # 4. Ensure we have at least one path (fallback)
        if not paths:
            fallback_path = self._create_fallback_path(start_lat, start_lon, end_lat, end_lon)
            paths.append(fallback_path)
        
        # Sort by preference (shortest first, then alternatives)
        paths = self._sort_paths_by_preference(paths)
        
        # Cache the results
        self._save_paths_to_cache(cache_key, paths)
        
        return paths[:max_paths]
    
    def _get_openrouteservice_alternatives(self, start_lat: float, start_lon: float, 
                                         end_lat: float, end_lon: float) -> List[PathInfo]:
        """Get alternative routes from OpenRouteService"""
        paths = []
        
        # Try different profiles
        profiles = ['cycling-regular', 'cycling-safe']
        
        for profile in profiles:
            try:
                url = f"https://api.openrouteservice.org/v2/directions/{profile}"
                
                params = {
                    'start': f"{start_lon},{start_lat}",
                    'end': f"{end_lon},{end_lat}",
                    'format': 'geojson',
                    'alternative_routes': 'true',
                    'alternative_routes.target_count': '3',
                    'alternative_routes.weight_factor': '1.4',
                    'alternative_routes.share_factor': '0.6'
                }
                
                # Add API key if available
                import os
                api_key = os.environ.get('OPENROUTESERVICE_API_KEY')
                headers = {'Authorization': api_key} if api_key else {}
                
                response = requests.get(url, params=params, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'features' in data:
                        for i, feature in enumerate(data['features']):
                            coordinates = feature['geometry']['coordinates']
                            properties = feature['properties']
                            
                            # Convert coordinates from [lon, lat] to [lat, lon]
                            path_coords = [(coord[1], coord[0]) for coord in coordinates]
                            
                            # Determine path type
                            if i == 0:
                                path_type = 'shortest' if profile == 'cycling-regular' else 'safest'
                            else:
                                path_type = f'alternative_{i}'
                            
                            # Extract route information
                            segments = properties.get('segments', [{}])
                            segment = segments[0] if segments else {}
                            
                            path_info = PathInfo(
                                path_id=f"ors_{profile}_{i}",
                                coordinates=path_coords,
                                distance_m=segment.get('distance', 0),
                                duration_s=segment.get('duration', 0),
                                path_type=path_type,
                                routing_profile=profile,
                                source='openrouteservice',
                                confidence=0.9 - (i * 0.1)  # Decrease confidence for alternatives
                            )
                            
                            paths.append(path_info)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"OpenRouteService {profile} failed: {e}")
                continue
        
        return paths
    
    def _get_osrm_alternatives(self, start_lat: float, start_lon: float, 
                             end_lat: float, end_lon: float) -> List[PathInfo]:
        """Get alternative routes from OSRM"""
        paths = []
        
        try:
            # OSRM with alternatives
            url = f"https://router.project-osrm.org/route/v1/bike/{start_lon},{start_lat};{end_lon},{end_lat}"
            
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'alternatives': 'true',
                'alternatives.max_paths': '3'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'routes' in data:
                    for i, route in enumerate(data['routes']):
                        coordinates = route['geometry']['coordinates']
                        
                        # Convert coordinates from [lon, lat] to [lat, lon]
                        path_coords = [(coord[1], coord[0]) for coord in coordinates]
                        
                        path_type = 'fastest' if i == 0 else f'alternative_{i + 2}'  # Offset to avoid conflicts
                        
                        path_info = PathInfo(
                            path_id=f"osrm_{i}",
                            coordinates=path_coords,
                            distance_m=route.get('distance', 0),
                            duration_s=route.get('duration', 0),
                            path_type=path_type,
                            routing_profile='cycling-regular',
                            source='osrm',
                            confidence=0.85 - (i * 0.1)
                        )
                        
                        paths.append(path_info)
        
        except Exception as e:
            logger.warning(f"OSRM alternatives failed: {e}")
        
        return paths
    
    def _generate_synthetic_alternatives(self, start_lat: float, start_lon: float, 
                                       end_lat: float, end_lon: float,
                                       existing_paths: List[PathInfo],
                                       needed_count: int) -> List[PathInfo]:
        """Generate synthetic alternative paths using waypoints"""
        paths = []
        
        if needed_count <= 0:
            return paths
        
        # Calculate midpoint and perpendicular offsets
        mid_lat = (start_lat + end_lat) / 2
        mid_lon = (start_lon + end_lon) / 2
        
        # Calculate bearing and perpendicular direction
        bearing = np.arctan2(end_lon - start_lon, end_lat - start_lat)
        perp_bearing = bearing + np.pi / 2
        
        # Generate waypoints for alternative routes
        waypoint_offsets = [0.002, -0.002, 0.004, -0.004]  # Different offset distances
        
        for i in range(min(needed_count, len(waypoint_offsets))):
            offset = waypoint_offsets[i]
            
            # Create waypoint perpendicular to direct route
            waypoint_lat = mid_lat + offset * np.cos(perp_bearing)
            waypoint_lon = mid_lon + offset * np.sin(perp_bearing)
            
            # Generate path through waypoint
            path_coords = self._generate_path_through_waypoint(
                start_lat, start_lon, waypoint_lat, waypoint_lon, end_lat, end_lon
            )
            
            # Calculate approximate distance and duration
            total_distance = 0
            for j in range(len(path_coords) - 1):
                total_distance += geodesic(path_coords[j], path_coords[j + 1]).meters
            
            duration = total_distance / 4.17  # Assume 15 km/h cycling speed
            
            path_info = PathInfo(
                path_id=f"synthetic_{i}",
                coordinates=path_coords,
                distance_m=total_distance,
                duration_s=duration,
                path_type=f'alternative_{len(existing_paths) + i + 1}',
                routing_profile='cycling-regular',
                source='synthetic',
                confidence=0.6 - (i * 0.1)
            )
            
            paths.append(path_info)
        
        return paths
    
    def _generate_path_through_waypoint(self, start_lat: float, start_lon: float,
                                      waypoint_lat: float, waypoint_lon: float,
                                      end_lat: float, end_lon: float) -> List[Tuple[float, float]]:
        """Generate a smooth path through a waypoint"""
        path_coords = []
        
        # First segment: start to waypoint
        segment1_points = 8
        for i in range(segment1_points):
            ratio = i / (segment1_points - 1)
            lat = start_lat + (waypoint_lat - start_lat) * ratio
            lon = start_lon + (waypoint_lon - start_lon) * ratio
            
            # Add some curvature
            if i > 0 and i < segment1_points - 1:
                curve_offset = 0.0005 * np.sin(ratio * np.pi)
                lat += curve_offset * np.random.uniform(-1, 1)
                lon += curve_offset * np.random.uniform(-1, 1)
            
            path_coords.append((lat, lon))
        
        # Second segment: waypoint to end
        segment2_points = 8
        for i in range(1, segment2_points):  # Skip first point to avoid duplication
            ratio = i / (segment2_points - 1)
            lat = waypoint_lat + (end_lat - waypoint_lat) * ratio
            lon = waypoint_lon + (end_lon - waypoint_lon) * ratio
            
            # Add some curvature
            if i > 0 and i < segment2_points - 1:
                curve_offset = 0.0005 * np.sin(ratio * np.pi)
                lat += curve_offset * np.random.uniform(-1, 1)
                lon += curve_offset * np.random.uniform(-1, 1)
            
            path_coords.append((lat, lon))
        
        return path_coords
    
    def _create_fallback_path(self, start_lat: float, start_lon: float, 
                            end_lat: float, end_lon: float) -> PathInfo:
        """Create a fallback path when all routing services fail"""
        
        # Create a simple path with intermediate points
        num_points = 6
        path_coords = []
        
        for i in range(num_points):
            ratio = i / (num_points - 1)
            lat = start_lat + (end_lat - start_lat) * ratio
            lon = start_lon + (end_lon - start_lon) * ratio
            
            # Add slight randomization for middle points
            if 0 < i < num_points - 1:
                lat += np.random.normal(0, 0.0002)
                lon += np.random.normal(0, 0.0002)
            
            path_coords.append((lat, lon))
        
        # Calculate distance
        total_distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).meters
        duration = total_distance / 4.17  # 15 km/h
        
        return PathInfo(
            path_id="fallback",
            coordinates=path_coords,
            distance_m=total_distance,
            duration_s=duration,
            path_type='shortest',
            routing_profile='cycling-regular',
            source='fallback',
            confidence=0.5
        )
    
    def _sort_paths_by_preference(self, paths: List[PathInfo]) -> List[PathInfo]:
        """Sort paths by preference (shortest first, then by confidence)"""
        
        # Define path type priority
        type_priority = {
            'shortest': 1,
            'fastest': 2,
            'safest': 3,
            'alternative_1': 4,
            'alternative_2': 5,
            'scenic': 6
        }
        
        # Sort by type priority, then by confidence
        return sorted(paths, key=lambda p: (
            type_priority.get(p.path_type, 99),
            -p.confidence,
            p.distance_m
        ))
    
    def _load_paths_from_cache(self, cache_key: str) -> Optional[List[PathInfo]]:
        """Load paths from cache"""
        import os
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                paths = []
                for item in data:
                    path_info = PathInfo(
                        path_id=item['path_id'],
                        coordinates=[(coord[0], coord[1]) for coord in item['coordinates']],
                        distance_m=item['distance_m'],
                        duration_s=item['duration_s'],
                        path_type=item['path_type'],
                        routing_profile=item['routing_profile'],
                        elevation_gain=item.get('elevation_gain', 0.0),
                        surface_types=item.get('surface_types', []),
                        traffic_level=item.get('traffic_level', 'unknown'),
                        bike_infrastructure=item.get('bike_infrastructure', 0.0),
                        confidence=item.get('confidence', 1.0),
                        source=item.get('source', 'unknown')
                    )
                    paths.append(path_info)
                
                return paths
                
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    def _save_paths_to_cache(self, cache_key: str, paths: List[PathInfo]):
        """Save paths to cache"""
        import os
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            data = []
            for path in paths:
                data.append({
                    'path_id': path.path_id,
                    'coordinates': path.coordinates,
                    'distance_m': path.distance_m,
                    'duration_s': path.duration_s,
                    'path_type': path.path_type,
                    'routing_profile': path.routing_profile,
                    'elevation_gain': path.elevation_gain,
                    'surface_types': path.surface_types,
                    'traffic_level': path.traffic_level,
                    'bike_infrastructure': path.bike_infrastructure,
                    'confidence': path.confidence,
                    'source': path.source
                })
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def analyze_path_diversity(self, paths: List[PathInfo]) -> Dict[str, float]:
        """Analyze the diversity of generated paths"""
        if len(paths) < 2:
            return {'diversity_score': 0.0, 'avg_distance_diff': 0.0, 'path_overlap': 1.0}
        
        # Calculate distance differences
        distances = [path.distance_m for path in paths]
        avg_distance_diff = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
        
        # Calculate path overlap (simplified - based on start/end point differences)
        overlaps = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                path1, path2 = paths[i], paths[j]
                
                # Sample points from each path for comparison
                sample_size = min(10, len(path1.coordinates), len(path2.coordinates))
                
                if sample_size > 1:
                    # Calculate average distance between corresponding points
                    total_distance = 0
                    for k in range(sample_size):
                        idx1 = int(k * (len(path1.coordinates) - 1) / (sample_size - 1))
                        idx2 = int(k * (len(path2.coordinates) - 1) / (sample_size - 1))
                        
                        coord1 = path1.coordinates[idx1]
                        coord2 = path2.coordinates[idx2]
                        
                        total_distance += geodesic(coord1, coord2).meters
                    
                    avg_distance = total_distance / sample_size
                    # Normalize by path length
                    path_length = (path1.distance_m + path2.distance_m) / 2
                    overlap = 1 - min(1, avg_distance / (path_length * 0.1))  # 10% threshold
                    overlaps.append(overlap)
        
        avg_overlap = np.mean(overlaps) if overlaps else 1.0
        
        # Diversity score combines distance variation and path separation
        diversity_score = (avg_distance_diff * 0.5) + ((1 - avg_overlap) * 0.5)
        
        return {
            'diversity_score': diversity_score,
            'avg_distance_diff': avg_distance_diff,
            'path_overlap': avg_overlap,
            'num_paths': len(paths)
        }
    
    def get_path_summary(self, paths: List[PathInfo]) -> pd.DataFrame:
        """Get a summary of all paths as a DataFrame"""
        data = []
        
        for path in paths:
            data.append({
                'path_id': path.path_id,
                'path_type': path.path_type,
                'distance_km': path.distance_m / 1000,
                'duration_min': path.duration_s / 60,
                'routing_profile': path.routing_profile,
                'confidence': path.confidence,
                'source': path.source,
                'speed_kmh': (path.distance_m / 1000) / (path.duration_s / 3600) if path.duration_s > 0 else 0
            })
        
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    router = MultiPathRouter()
    
    # Example coordinates (Lausanne area)
    start_lat, start_lon = 46.5197, 6.6323
    end_lat, end_lon = 46.5238, 6.6356
    
    print("Generating multiple paths...")
    paths = router.get_multiple_paths(start_lat, start_lon, end_lat, end_lon, max_paths=5)
    
    print(f"\nGenerated {len(paths)} paths:")
    for i, path in enumerate(paths, 1):
        print(f"{i}. {path.path_type} ({path.source}): {path.distance_m:.0f}m, {path.duration_s/60:.1f}min, confidence: {path.confidence:.2f}")
    
    # Analyze diversity
    diversity = router.analyze_path_diversity(paths)
    print(f"\nPath diversity analysis:")
    for key, value in diversity.items():
        print(f"{key}: {value:.3f}")
    
    # Get summary
    summary = router.get_path_summary(paths)
    print(f"\nPath summary:")
    print(summary.to_string(index=False))