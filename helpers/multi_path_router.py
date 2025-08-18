"""
Improved Multi-Path Router for Bike Flow Prediction
Prioritizes real road-following routing services to avoid paths through buildings.
Uses multiple routing APIs and intelligent fallbacks.
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
    follows_roads: bool = True  # Whether this path follows actual roads

class ImprovedMultiPathRouter:
    """Generate multiple routing options that follow actual roads and bike paths"""
    
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
                'description': 'Standard cycling route following roads',
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
                'description': 'Fastest cycling route on roads',
                'preference': 'speed',
                'avoid': [],
                'priority': 'time'
            },
            'cycling-scenic': {
                'description': 'Scenic route through parks and quiet roads',
                'preference': 'scenery',
                'avoid': ['highways', 'industrial'],
                'priority': 'experience'
            }
        }
    
    def _define_path_types(self) -> Dict[str, Dict]:
        """Define different types of paths to generate"""
        return {
            'shortest': {
                'description': 'Shortest distance path on roads',
                'weight': 'distance',
                'alternatives': False
            },
            'fastest': {
                'description': 'Fastest time path on roads',
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
        """Get multiple path options that follow actual roads"""
        
        # Generate cache key
        cache_key = f"improved_{start_lat:.6f}_{start_lon:.6f}_{end_lat:.6f}_{end_lon:.6f}_{max_paths}"
        
        # Try to load from cache
        cached_paths = self._load_paths_from_cache(cache_key)
        if cached_paths:
            return cached_paths[:max_paths]
        
        paths = []
        
        # 1. Try OSRM for cycling routes (FREE, RELIABLE)
        osrm_paths = self._get_osrm_cycling_routes(start_lat, start_lon, end_lat, end_lon)
        paths.extend(osrm_paths)
        
        # 2. Try GraphHopper (FREE tier available)
        if len(paths) < max_paths:
            graphhopper_paths = self._get_graphhopper_routes(start_lat, start_lon, end_lat, end_lon)
            paths.extend(graphhopper_paths)
        
        # 3. Generate road-following alternatives using waypoints
        if len(paths) < max_paths:
            waypoint_paths = self._generate_waypoint_road_routes(
                start_lat, start_lon, end_lat, end_lon, 
                existing_paths=paths,
                needed_count=max_paths - len(paths)
            )
            paths.extend(waypoint_paths)
        
        # 4. Ensure we have at least one path (improved fallback that follows roads)
        if not paths:
            fallback_path = self._create_road_following_fallback(start_lat, start_lon, end_lat, end_lon)
            paths.append(fallback_path)
        
        # Remove duplicates and sort by preference
        paths = self._remove_duplicate_paths(paths)
        paths = self._sort_paths_by_preference(paths)
        
        # Cache the results
        self._save_paths_to_cache(cache_key, paths)
        
        return paths[:max_paths]
    
    def _get_osrm_cycling_routes(self, start_lat: float, start_lon: float, 
                                end_lat: float, end_lon: float) -> List[PathInfo]:
        """Get cycling routes from OSRM (follows actual roads)"""
        paths = []
        
        try:
            # OSRM cycling profile with alternatives
            url = f"https://router.project-osrm.org/route/v1/bike/{start_lon},{start_lat};{end_lon},{end_lat}"
            
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'alternatives': 'true',
                'alternatives.max_paths': '3',
                'steps': 'false'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'routes' in data and len(data['routes']) > 0:
                    for i, route in enumerate(data['routes']):
                        if 'geometry' in route and 'coordinates' in route['geometry']:
                            coordinates = route['geometry']['coordinates']
                            
                            # Convert coordinates from [lon, lat] to [lat, lon]
                            path_coords = [(coord[1], coord[0]) for coord in coordinates]
                            
                            # Determine path type
                            if i == 0:
                                path_type = 'fastest'
                            else:
                                path_type = f'alternative_{i}'
                            
                            path_info = PathInfo(
                                path_id=f"osrm_bike_{i}",
                                coordinates=path_coords,
                                distance_m=route.get('distance', 0),
                                duration_s=route.get('duration', 0),
                                path_type=path_type,
                                routing_profile='cycling-regular',
                                source='osrm_bike',
                                confidence=0.95 - (i * 0.05),  # High confidence for real roads
                                follows_roads=True
                            )
                            
                            paths.append(path_info)
                            logger.info(f"OSRM: Generated {path_type} path with {len(path_coords)} points")
        
        except Exception as e:
            logger.warning(f"OSRM cycling routes failed: {e}")
        
        return paths
    
    def _get_openrouteservice_routes(self, start_lat: float, start_lon: float, 
                                   end_lat: float, end_lon: float) -> List[PathInfo]:
        """Get cycling routes from OpenRouteService"""
        paths = []
        
        # Check for API key
        import os
        api_key = os.environ.get('OPENROUTESERVICE_API_KEY')
        if not api_key:
            logger.info("OpenRouteService API key not found, skipping")
            return paths
        
        profiles = ['cycling-regular', 'cycling-safe']
        
        for profile in profiles:
            try:
                url = f"https://api.openrouteservice.org/v2/directions/{profile}"
                
                params = {
                    'start': f"{start_lon},{start_lat}",
                    'end': f"{end_lon},{end_lat}",
                    'format': 'geojson',
                    'alternative_routes': 'true',
                    'alternative_routes.target_count': '2',
                    'alternative_routes.weight_factor': '1.4',
                    'alternative_routes.share_factor': '0.6'
                }
                
                headers = {'Authorization': api_key}
                
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
                                path_type = f'alternative_{i + 10}'  # Offset to avoid conflicts
                            
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
                                confidence=0.9 - (i * 0.05),
                                follows_roads=True
                            )
                            
                            paths.append(path_info)
                            logger.info(f"ORS: Generated {path_type} path with {len(path_coords)} points")
                
                # Rate limiting
                time.sleep(1.0)
                
            except Exception as e:
                logger.warning(f"OpenRouteService {profile} failed: {e}")
                continue
        
        return paths
    
    def _get_graphhopper_routes(self, start_lat: float, start_lon: float, 
                              end_lat: float, end_lon: float) -> List[PathInfo]:
        """Get cycling routes from GraphHopper"""
        paths = []
        
        try:
            # GraphHopper cycling profile
            url = "https://graphhopper.com/api/1/route"
            
            params = {
                'point': [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
                'vehicle': 'bike',
                'locale': 'en',
                'calc_points': 'true',
                'debug': 'false',
                'elevation': 'false',
                'points_encoded': 'false',
                'type': 'json',
                'alternative_route.max_paths': '2',
                'alternative_route.max_weight_factor': '1.4'
            }
            
            # Add API key if available
            import os
            api_key = os.environ.get('GRAPHHOPPER_API_KEY')
            if api_key:
                params['key'] = api_key
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'paths' in data:
                    for i, path_data in enumerate(data['paths']):
                        if 'points' in path_data and 'coordinates' in path_data['points']:
                            coordinates = path_data['points']['coordinates']
                            
                            # Convert coordinates from [lon, lat] to [lat, lon]
                            path_coords = [(coord[1], coord[0]) for coord in coordinates]
                            
                            path_type = 'shortest' if i == 0 else f'alternative_{i + 20}'
                            
                            path_info = PathInfo(
                                path_id=f"graphhopper_{i}",
                                coordinates=path_coords,
                                distance_m=path_data.get('distance', 0),
                                duration_s=path_data.get('time', 0) / 1000,  # Convert from ms
                                path_type=path_type,
                                routing_profile='cycling-regular',
                                source='graphhopper',
                                confidence=0.85 - (i * 0.05),
                                follows_roads=True
                            )
                            
                            paths.append(path_info)
                            logger.info(f"GraphHopper: Generated {path_type} path with {len(path_coords)} points")
        
        except Exception as e:
            logger.warning(f"GraphHopper routes failed: {e}")
        
        return paths
    
    def _get_mapbox_routes(self, start_lat: float, start_lon: float, 
                         end_lat: float, end_lon: float) -> List[PathInfo]:
        """Get cycling routes from Mapbox"""
        paths = []
        
        # Check for API key
        import os
        api_key = os.environ.get('MAPBOX_API_KEY')
        if not api_key:
            logger.info("Mapbox API key not found, skipping")
            return paths
        
        try:
            # Mapbox cycling profile
            url = f"https://api.mapbox.com/directions/v5/mapbox/cycling/{start_lon},{start_lat};{end_lon},{end_lat}"
            
            params = {
                'alternatives': 'true',
                'geometries': 'geojson',
                'overview': 'full',
                'steps': 'false',
                'access_token': api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'routes' in data:
                    for i, route in enumerate(data['routes']):
                        if 'geometry' in route and 'coordinates' in route['geometry']:
                            coordinates = route['geometry']['coordinates']
                            
                            # Convert coordinates from [lon, lat] to [lat, lon]
                            path_coords = [(coord[1], coord[0]) for coord in coordinates]
                            
                            path_type = 'fastest' if i == 0 else f'alternative_{i + 30}'
                            
                            path_info = PathInfo(
                                path_id=f"mapbox_{i}",
                                coordinates=path_coords,
                                distance_m=route.get('distance', 0),
                                duration_s=route.get('duration', 0),
                                path_type=path_type,
                                routing_profile='cycling-regular',
                                source='mapbox',
                                confidence=0.9 - (i * 0.05),
                                follows_roads=True
                            )
                            
                            paths.append(path_info)
                            logger.info(f"Mapbox: Generated {path_type} path with {len(path_coords)} points")
        
        except Exception as e:
            logger.warning(f"Mapbox routes failed: {e}")
        
        return paths
    
    def _generate_waypoint_road_routes(self, start_lat: float, start_lon: float, 
                                     end_lat: float, end_lon: float,
                                     existing_paths: List[PathInfo],
                                     needed_count: int) -> List[PathInfo]:
        """Generate alternative routes using waypoints and routing services"""
        paths = []
        
        if needed_count <= 0:
            return paths
        
        # Generate strategic waypoints that might lead to different roads
        waypoints = self._generate_strategic_waypoints(start_lat, start_lon, end_lat, end_lon, needed_count)
        
        for i, waypoint in enumerate(waypoints):
            try:
                # Try to route through the waypoint using OSRM
                waypoint_path = self._route_through_waypoint(
                    start_lat, start_lon, waypoint[0], waypoint[1], end_lat, end_lon
                )
                
                if waypoint_path:
                    waypoint_path.path_id = f"waypoint_{i}"
                    waypoint_path.path_type = f'alternative_{len(existing_paths) + i + 40}'
                    waypoint_path.source = 'waypoint_routing'
                    waypoint_path.confidence = 0.7 - (i * 0.1)
                    paths.append(waypoint_path)
                    
            except Exception as e:
                logger.warning(f"Waypoint routing {i} failed: {e}")
                continue
        
        return paths
    
    def _generate_strategic_waypoints(self, start_lat: float, start_lon: float, 
                                    end_lat: float, end_lon: float, count: int) -> List[Tuple[float, float]]:
        """Generate strategic waypoints that might lead to different road routes"""
        waypoints = []
        
        # Calculate midpoint
        mid_lat = (start_lat + end_lat) / 2
        mid_lon = (start_lon + end_lon) / 2
        
        # Calculate perpendicular direction
        bearing = np.arctan2(end_lon - start_lon, end_lat - start_lat)
        perp_bearing = bearing + np.pi / 2
        
        # Generate waypoints at different offsets and positions
        offsets = [0.003, -0.003, 0.005, -0.005]  # Larger offsets for more distinct routes
        positions = [0.3, 0.7, 0.4, 0.6]  # Different positions along the route
        
        for i in range(min(count, len(offsets))):
            # Position along the direct route
            position = positions[i % len(positions)]
            base_lat = start_lat + (end_lat - start_lat) * position
            base_lon = start_lon + (end_lon - start_lon) * position
            
            # Offset perpendicular to the route
            offset = offsets[i]
            waypoint_lat = base_lat + offset * np.cos(perp_bearing)
            waypoint_lon = base_lon + offset * np.sin(perp_bearing)
            
            waypoints.append((waypoint_lat, waypoint_lon))
        
        return waypoints
    
    def _route_through_waypoint(self, start_lat: float, start_lon: float,
                              waypoint_lat: float, waypoint_lon: float,
                              end_lat: float, end_lon: float) -> Optional[PathInfo]:
        """Route through a waypoint using OSRM"""
        try:
            # Route from start to waypoint to end
            url = f"https://router.project-osrm.org/route/v1/bike/{start_lon},{start_lat};{waypoint_lon},{waypoint_lat};{end_lon},{end_lat}"
            
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'false'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'routes' in data and len(data['routes']) > 0:
                    route = data['routes'][0]
                    
                    if 'geometry' in route and 'coordinates' in route['geometry']:
                        coordinates = route['geometry']['coordinates']
                        
                        # Convert coordinates from [lon, lat] to [lat, lon]
                        path_coords = [(coord[1], coord[0]) for coord in coordinates]
                        
                        path_info = PathInfo(
                            path_id="waypoint_temp",
                            coordinates=path_coords,
                            distance_m=route.get('distance', 0),
                            duration_s=route.get('duration', 0),
                            path_type='alternative',
                            routing_profile='cycling-regular',
                            source='waypoint_routing',
                            confidence=0.7,
                            follows_roads=True
                        )
                        
                        return path_info
        
        except Exception as e:
            logger.warning(f"Waypoint routing failed: {e}")
        
        return None
    
    def _create_road_following_fallback(self, start_lat: float, start_lon: float, 
                                      end_lat: float, end_lon: float) -> PathInfo:
        """Create a fallback path that attempts to follow roads using grid approximation"""
        
        # Try a simple grid-based approach to approximate road following
        path_coords = self._generate_grid_based_path(start_lat, start_lon, end_lat, end_lon)
        
        # Calculate distance along the path
        total_distance = 0
        for i in range(len(path_coords) - 1):
            total_distance += geodesic(path_coords[i], path_coords[i + 1]).meters
        
        duration = total_distance / 4.17  # 15 km/h cycling speed
        
        return PathInfo(
            path_id="grid_fallback",
            coordinates=path_coords,
            distance_m=total_distance,
            duration_s=duration,
            path_type='shortest',
            routing_profile='cycling-regular',
            source='grid_fallback',
            confidence=0.5,
            follows_roads=False  # This is still an approximation
        )
    
    def _generate_grid_based_path(self, start_lat: float, start_lon: float, 
                                end_lat: float, end_lon: float) -> List[Tuple[float, float]]:
        """Generate a path that follows a grid pattern (approximating city streets)"""
        path_coords = []
        
        # Calculate differences
        lat_diff = end_lat - start_lat
        lon_diff = end_lon - start_lon
        
        # Determine if we should go more north-south or east-west first
        if abs(lat_diff) > abs(lon_diff):
            # Go north-south first, then east-west
            mid_lat = end_lat
            mid_lon = start_lon
        else:
            # Go east-west first, then north-south
            mid_lat = start_lat
            mid_lon = end_lon
        
        # Create path: start -> intermediate corner -> end
        path_coords.append((start_lat, start_lon))
        
        # Add intermediate points along the first leg
        num_intermediate = 3
        for i in range(1, num_intermediate):
            ratio = i / num_intermediate
            if abs(lat_diff) > abs(lon_diff):
                # Moving north-south first
                intermediate_lat = start_lat + lat_diff * ratio
                intermediate_lon = start_lon
            else:
                # Moving east-west first
                intermediate_lat = start_lat
                intermediate_lon = start_lon + lon_diff * ratio
            
            path_coords.append((intermediate_lat, intermediate_lon))
        
        # Add the corner point
        path_coords.append((mid_lat, mid_lon))
        
        # Add intermediate points along the second leg
        for i in range(1, num_intermediate):
            ratio = i / num_intermediate
            if abs(lat_diff) > abs(lon_diff):
                # Moving east-west second
                intermediate_lat = mid_lat
                intermediate_lon = mid_lon + (end_lon - mid_lon) * ratio
            else:
                # Moving north-south second
                intermediate_lat = mid_lat + (end_lat - mid_lat) * ratio
                intermediate_lon = mid_lon
            
            path_coords.append((intermediate_lat, intermediate_lon))
        
        # Add the end point
        path_coords.append((end_lat, end_lon))
        
        return path_coords
    
    def _remove_duplicate_paths(self, paths: List[PathInfo]) -> List[PathInfo]:
        """Remove duplicate or very similar paths"""
        if len(paths) <= 1:
            return paths
        
        unique_paths = []
        
        for path in paths:
            is_duplicate = False
            
            for existing_path in unique_paths:
                # Check if paths are very similar
                if self._paths_are_similar(path, existing_path):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if path.confidence > existing_path.confidence:
                        unique_paths.remove(existing_path)
                        unique_paths.append(path)
                    break
            
            if not is_duplicate:
                unique_paths.append(path)
        
        return unique_paths
    
    def _paths_are_similar(self, path1: PathInfo, path2: PathInfo, threshold: float = 0.1) -> bool:
        """Check if two paths are very similar"""
        # Check distance similarity
        if abs(path1.distance_m - path2.distance_m) / max(path1.distance_m, path2.distance_m) < threshold:
            # Check coordinate similarity (sample a few points)
            if len(path1.coordinates) > 2 and len(path2.coordinates) > 2:
                # Compare start, middle, and end points
                start_dist = geodesic(path1.coordinates[0], path2.coordinates[0]).meters
                end_dist = geodesic(path1.coordinates[-1], path2.coordinates[-1]).meters
                
                mid1_idx = len(path1.coordinates) // 2
                mid2_idx = len(path2.coordinates) // 2
                mid_dist = geodesic(path1.coordinates[mid1_idx], path2.coordinates[mid2_idx]).meters
                
                avg_dist = (start_dist + end_dist + mid_dist) / 3
                
                # If average distance between corresponding points is small, they're similar
                return avg_dist < 100  # 100 meters threshold
        
        return False
    
    def _sort_paths_by_preference(self, paths: List[PathInfo]) -> List[PathInfo]:
        """Sort paths by preference (real roads first, then by type and confidence)"""
        
        # Define path type priority
        type_priority = {
            'shortest': 1,
            'fastest': 2,
            'safest': 3,
            'alternative_1': 4,
            'alternative_2': 5,
            'scenic': 6
        }
        
        # Sort by: follows_roads (True first), type priority, confidence, distance
        return sorted(paths, key=lambda p: (
            not p.follows_roads,  # False (follows roads) comes first
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
                        source=item.get('source', 'unknown'),
                        follows_roads=item.get('follows_roads', True)
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
                    'source': path.source,
                    'follows_roads': path.follows_roads
                })
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def analyze_path_quality(self, paths: List[PathInfo]) -> Dict[str, float]:
        """Analyze the quality of generated paths"""
        if not paths:
            return {'road_following_ratio': 0.0, 'avg_confidence': 0.0, 'diversity_score': 0.0}
        
        # Calculate road-following ratio
        road_following_count = sum(1 for path in paths if path.follows_roads)
        road_following_ratio = road_following_count / len(paths)
        
        # Calculate average confidence
        avg_confidence = np.mean([path.confidence for path in paths])
        
        # Calculate diversity (distance variation)
        if len(paths) > 1:
            distances = [path.distance_m for path in paths]
            diversity_score = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
        else:
            diversity_score = 0.0
        
        return {
            'road_following_ratio': road_following_ratio,
            'avg_confidence': avg_confidence,
            'diversity_score': diversity_score,
            'num_paths': len(paths),
            'real_road_paths': road_following_count
        }

if __name__ == "__main__":
    # Example usage
    router = ImprovedMultiPathRouter()
    
    # Example coordinates (Lausanne area)
    start_lat, start_lon = 46.5197, 6.6323
    end_lat, end_lon = 46.5238, 6.6356
    
    print("🚴‍♂️ Generating improved road-following paths...")
    paths = router.get_multiple_paths(start_lat, start_lon, end_lat, end_lon, max_paths=5)
    
    print(f"\n📊 Generated {len(paths)} paths:")
    for i, path in enumerate(paths, 1):
        road_status = "🛣️ " if path.follows_roads else "⚠️ "
        print(f"{i}. {road_status}{path.path_type} ({path.source}): {path.distance_m:.0f}m, {path.duration_s/60:.1f}min, confidence: {path.confidence:.2f}")
    
    # Analyze quality
    quality = router.analyze_path_quality(paths)
    print(f"\n🎯 Path Quality Analysis:")
    for key, value in quality.items():
        print(f"{key}: {value:.3f}")