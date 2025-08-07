"""
Multi-Stop Journey Predictor for Bike Flow Analysis
Predicts intermediate POIs (banks, restaurants, schools, etc.) that cyclists visit
between origin and destination stations, creating realistic journey patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from geopy.distance import geodesic
import json
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
import time
from datetime import datetime
import requests
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class POI:
    """Point of Interest with enriched data"""
    poi_id: str
    poi_type: str  # 'bank', 'restaurant', 'school', etc.
    lat: float
    lon: float
    name: str
    tags: Dict
    visit_probability: float = 0.0
    visit_duration_minutes: float = 0.0
    popularity_score: float = 0.0
    accessibility_score: float = 0.0

@dataclass
class JourneySegment:
    """A segment of a multi-stop journey"""
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    poi: Optional[POI] = None
    distance_m: float = 0.0
    duration_minutes: float = 0.0
    path_coordinates: List[Tuple[float, float]] = None
    segment_type: str = 'direct'  # 'direct', 'via_poi', 'detour'

@dataclass
class MultiStopJourney:
    """Complete journey with multiple stops"""
    journey_id: str
    origin_station: str
    destination_station: str
    segments: List[JourneySegment]
    total_distance_m: float
    total_duration_minutes: float
    visit_probability: float
    journey_type: str  # 'direct', 'shopping', 'errands', 'leisure', 'work'
    time_of_day: int
    day_of_week: int
    population_context: Dict[str, float]

class MultiStopJourneyPredictor:
    """Predicts multi-stop journeys with POI visits"""
    
    def __init__(self, osm_extractor, population_extractor, multi_router, cache_dir: str = "cache/journeys"):
        self.osm_extractor = osm_extractor
        self.population_extractor = population_extractor
        self.multi_router = multi_router
        self.cache_dir = cache_dir
        
        # Journey prediction models
        self.poi_visit_model = None
        self.journey_type_model = None
        self.duration_model = None
        
        # POI database
        self.poi_database = {}
        self.poi_graph = nx.Graph()
        
        # Journey patterns
        self.journey_patterns = {}
        self.temporal_patterns = {}
        
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def build_poi_database(self, station_coords: Dict[str, Tuple[float, float]], radius_m: int = 1000):
        """Build comprehensive POI database from OSM features around all stations"""
        logger.info("Building comprehensive POI database...")
        
        all_pois = {}
        station_coverage = {}
        
        for station_id, (lat, lon) in station_coords.items():
            try:
                # Extract OSM features
                osm_features = self.osm_extractor.extract_features_around_station(lat, lon, radius_m)
                
                # Extract population context
                pop_features = self.population_extractor.extract_population_features_around_station(lat, lon, radius_m)
                
                station_pois = []
                
                # Convert OSM features to POIs
                for poi_type, poi_list in osm_features.items():
                    for poi_data in poi_list:
                        poi_id = f"{poi_type}_{poi_data.lat:.6f}_{poi_data.lon:.6f}"
                        
                        if poi_id not in all_pois:
                            # Calculate POI characteristics
                            popularity = self._calculate_poi_popularity(poi_data, pop_features)
                            accessibility = self._calculate_poi_accessibility(poi_data, station_coords)
                            
                            poi = POI(
                                poi_id=poi_id,
                                poi_type=poi_type,
                                lat=poi_data.lat,
                                lon=poi_data.lon,
                                name=poi_data.tags.get('name', f'{poi_type}_{poi_id[-8:]}'),
                                tags=poi_data.tags,
                                popularity_score=popularity,
                                accessibility_score=accessibility
                            )
                            
                            all_pois[poi_id] = poi
                        
                        station_pois.append(poi_id)
                
                station_coverage[station_id] = {
                    'pois': station_pois,
                    'population': pop_features,
                    'coordinates': (lat, lon)
                }
                
                if len(all_pois) % 100 == 0:
                    logger.info(f"Built {len(all_pois)} POIs so far...")
                    
            except Exception as e:
                logger.warning(f"Failed to process station {station_id}: {e}")
                continue
        
        self.poi_database = all_pois
        self.station_coverage = station_coverage
        
        # Build POI graph for routing
        self._build_poi_graph()
        
        logger.info(f"Built POI database with {len(all_pois)} POIs covering {len(station_coverage)} stations")
        
        return all_pois
    
    def _calculate_poi_popularity(self, poi_data, pop_features: Dict[str, float]) -> float:
        """Calculate POI popularity score based on population and type"""
        base_popularity = {
            'restaurants': 0.8,
            'cafes': 0.7,
            'shops': 0.6,
            'supermarkets': 0.9,
            'banks': 0.5,
            'atms': 0.4,
            'schools': 0.7,
            'parks': 0.6,
            'hospitals': 0.8,
            'pharmacies': 0.7,
            'bus_stops': 0.9,
            'train_stations': 0.95
        }.get(poi_data.feature_type, 0.3)
        
        # Adjust by population density
        pop_density = pop_features.get('population_density_km2', 0)
        pop_factor = min(2.0, 1.0 + (pop_density / 5000))  # Higher density = more popular
        
        # Adjust by distance to population center
        pop_center_dist = pop_features.get('population_center_distance', 1000)
        distance_factor = max(0.3, 1.0 - (pop_center_dist / 1000))  # Closer to pop center = more popular
        
        return min(1.0, base_popularity * pop_factor * distance_factor)
    
    def _calculate_poi_accessibility(self, poi_data, station_coords: Dict[str, Tuple[float, float]]) -> float:
        """Calculate POI accessibility score based on proximity to bike stations"""
        min_distance = float('inf')
        
        for station_id, (station_lat, station_lon) in station_coords.items():
            distance = geodesic((poi_data.lat, poi_data.lon), (station_lat, station_lon)).meters
            min_distance = min(min_distance, distance)
        
        # Convert to accessibility score (closer = more accessible)
        if min_distance < 100:
            return 1.0
        elif min_distance < 250:
            return 0.8
        elif min_distance < 500:
            return 0.6
        elif min_distance < 1000:
            return 0.4
        else:
            return 0.2
    
    def _build_poi_graph(self):
        """Build a graph connecting POIs for routing"""
        logger.info("Building POI connectivity graph...")
        
        self.poi_graph = nx.Graph()
        
        # Add all POIs as nodes
        for poi_id, poi in self.poi_database.items():
            self.poi_graph.add_node(poi_id, 
                                  lat=poi.lat, 
                                  lon=poi.lon, 
                                  poi_type=poi.poi_type,
                                  popularity=poi.popularity_score)
        
        # Connect nearby POIs (within walking/cycling distance)
        poi_list = list(self.poi_database.items())
        
        for i, (poi1_id, poi1) in enumerate(poi_list):
            for j, (poi2_id, poi2) in enumerate(poi_list[i+1:], i+1):
                distance = geodesic((poi1.lat, poi1.lon), (poi2.lat, poi2.lon)).meters
                
                # Connect POIs within reasonable cycling distance
                if distance <= 2000:  # 2km max
                    travel_time = distance / 250 * 60  # ~15 km/h cycling speed in minutes
                    
                    self.poi_graph.add_edge(poi1_id, poi2_id, 
                                          distance=distance, 
                                          travel_time=travel_time)
        
        logger.info(f"Built POI graph with {len(self.poi_graph.nodes)} nodes and {len(self.poi_graph.edges)} edges")
    
    def learn_journey_patterns(self, trips_df: pd.DataFrame, station_coords: Dict[str, Tuple[float, float]]):
        """Learn journey patterns from historical trip data"""
        logger.info("Learning journey patterns from trip data...")
        
        # Analyze trip patterns by time and purpose
        temporal_analysis = self._analyze_temporal_patterns(trips_df)
        spatial_analysis = self._analyze_spatial_patterns(trips_df, station_coords)
        
        # Train ML models for different aspects
        self._train_poi_visit_models(trips_df, station_coords)
        self._train_journey_type_models(trips_df)
        self._train_duration_models(trips_df)
        
        self.temporal_patterns = temporal_analysis
        self.spatial_patterns = spatial_analysis
        
        logger.info("Journey pattern learning completed")
    
    def _analyze_temporal_patterns(self, trips_df: pd.DataFrame) -> Dict:
        """Analyze when people make different types of journeys"""
        patterns = {}
        
        # Hour-based patterns
        hourly_patterns = trips_df.groupby('hour').agg({
            'trip_id': 'count',
            'duration': 'mean'
        }).to_dict('index')
        
        # Day of week patterns
        dow_patterns = trips_df.groupby('day_of_week').agg({
            'trip_id': 'count',
            'duration': 'mean'
        }).to_dict('index')
        
        # Define journey types by time patterns
        patterns['rush_hour_morning'] = [7, 8, 9]  # Work commute
        patterns['lunch_time'] = [12, 13]  # Lunch trips
        patterns['evening_rush'] = [17, 18, 19]  # Evening commute
        patterns['leisure_time'] = [10, 11, 14, 15, 16]  # Shopping, errands
        patterns['evening_leisure'] = [20, 21, 22]  # Dining, entertainment
        
        patterns['hourly'] = hourly_patterns
        patterns['daily'] = dow_patterns
        
        return patterns
    
    def _analyze_spatial_patterns(self, trips_df: pd.DataFrame, station_coords: Dict) -> Dict:
        """Analyze spatial journey patterns"""
        patterns = {}
        
        # Calculate common routes
        route_counts = trips_df.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='count')
        route_counts = route_counts.sort_values('count', ascending=False)
        
        # Identify journey types by distance and direction
        patterns['short_trips'] = []  # < 1km
        patterns['medium_trips'] = []  # 1-3km
        patterns['long_trips'] = []  # > 3km
        
        for _, row in route_counts.iterrows():
            start_id, end_id = row['start_station_id'], row['end_station_id']
            
            if start_id in station_coords and end_id in station_coords:
                start_coord = station_coords[start_id]
                end_coord = station_coords[end_id]
                distance = geodesic(start_coord, end_coord).meters
                
                trip_info = {
                    'start': start_id,
                    'end': end_id,
                    'distance': distance,
                    'frequency': row['count']
                }
                
                if distance < 1000:
                    patterns['short_trips'].append(trip_info)
                elif distance < 3000:
                    patterns['medium_trips'].append(trip_info)
                else:
                    patterns['long_trips'].append(trip_info)
        
        return patterns
    
    def _train_poi_visit_models(self, trips_df: pd.DataFrame, station_coords: Dict):
        """Train models to predict POI visit probability"""
        # This is a simplified training - in reality you'd need more detailed data
        # For now, we'll use heuristics based on time patterns and POI types
        
        # Create training data based on journey characteristics
        training_data = []
        
        # Sample some trips for training
        sample_trips = trips_df.sample(n=min(1000, len(trips_df)), random_state=42)
        
        for _, trip in sample_trips.iterrows():
            start_id = trip['start_station_id']
            end_id = trip['end_station_id']
            hour = trip['hour']
            dow = trip['day_of_week']
            
            if start_id in station_coords and end_id in station_coords:
                # Get POIs along the route
                route_pois = self._get_pois_along_route(
                    station_coords[start_id], 
                    station_coords[end_id]
                )
                
                # Create features for POI visit prediction
                for poi in route_pois:
                    features = self._create_poi_features(poi, hour, dow, trip)
                    
                    # Simulate visit probability (in real system, this would come from actual data)
                    visit_prob = self._simulate_visit_probability(poi, hour, dow)
                    
                    training_data.append(features + [visit_prob])
        
        if training_data:
            training_data = np.array(training_data)
            X = training_data[:, :-1]
            y = training_data[:, -1]
            
            self.poi_visit_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.poi_visit_model.fit(X, y)
            
            logger.info("POI visit model trained successfully")
    
    def _get_pois_along_route(self, start_coord: Tuple[float, float], 
                            end_coord: Tuple[float, float], 
                            buffer_m: int = 500) -> List[POI]:
        """Get POIs along a route within a buffer distance"""
        route_pois = []
        
        # Simple implementation: get POIs within buffer of straight line
        for poi_id, poi in self.poi_database.items():
            # Calculate distance from POI to line segment
            dist_to_route = self._point_to_line_distance(
                (poi.lat, poi.lon), start_coord, end_coord
            )
            
            if dist_to_route <= buffer_m:
                route_pois.append(poi)
        
        return route_pois
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment"""
        # Simplified calculation using geodesic distances
        dist_to_start = geodesic(point, line_start).meters
        dist_to_end = geodesic(point, line_end).meters
        line_length = geodesic(line_start, line_end).meters
        
        if line_length == 0:
            return dist_to_start
        
        # Approximate distance to line
        # This is a simplification - for more accuracy, use proper geometric calculations
        return min(dist_to_start, dist_to_end, (dist_to_start + dist_to_end - line_length) / 2)
    
    def _create_poi_features(self, poi: POI, hour: int, dow: int, trip_data) -> List[float]:
        """Create feature vector for POI visit prediction"""
        return [
            hour,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            dow,
            1 if dow in [5, 6] else 0,  # weekend
            poi.popularity_score,
            poi.accessibility_score,
            self._get_poi_type_encoding(poi.poi_type),
            trip_data.get('duration', 0) / 60,  # trip duration in hours
            trip_data.get('temperature', 20)  # weather
        ]
    
    def _get_poi_type_encoding(self, poi_type: str) -> float:
        """Encode POI type as numerical value"""
        encoding = {
            'restaurants': 0.9,
            'cafes': 0.8,
            'shops': 0.7,
            'supermarkets': 0.8,
            'banks': 0.5,
            'atms': 0.4,
            'schools': 0.6,
            'parks': 0.7,
            'hospitals': 0.9,
            'pharmacies': 0.6,
            'bus_stops': 0.3,
            'train_stations': 0.4
        }
        return encoding.get(poi_type, 0.3)
    
    def _simulate_visit_probability(self, poi: POI, hour: int, dow: int) -> float:
        """Simulate POI visit probability (replace with real data learning)"""
        base_prob = poi.popularity_score * 0.3
        
        # Time-based adjustments
        if poi.poi_type in ['restaurants', 'cafes']:
            if hour in [12, 13, 19, 20]:  # Meal times
                base_prob *= 2.0
        elif poi.poi_type in ['banks', 'shops']:
            if 9 <= hour <= 17 and dow < 5:  # Business hours, weekdays
                base_prob *= 1.5
        elif poi.poi_type in ['bars', 'nightclubs']:
            if hour >= 20:  # Evening
                base_prob *= 2.0
        
        return min(1.0, base_prob)
    
    def _train_journey_type_models(self, trips_df: pd.DataFrame):
        """Train models to predict journey types"""
        # Simplified journey type classification
        self.journey_type_model = "heuristic"  # Placeholder
        logger.info("Journey type model initialized")
    
    def _train_duration_models(self, trips_df: pd.DataFrame):
        """Train models to predict visit durations"""
        # Simplified duration model
        self.duration_model = "heuristic"  # Placeholder
        logger.info("Duration model initialized")
    
    def predict_multi_stop_journey(self, start_station: str, end_station: str, 
                                  hour: int, day_of_week: int,
                                  max_stops: int = 3,
                                  max_detour_factor: float = 2.0) -> List[MultiStopJourney]:
        """Predict multi-stop journeys with POI visits"""
        
        if start_station not in self.station_coverage or end_station not in self.station_coverage:
            return []
        
        start_coord = self.station_coverage[start_station]['coordinates']
        end_coord = self.station_coverage[end_station]['coordinates']
        
        # Get direct distance for detour calculation
        direct_distance = geodesic(start_coord, end_coord).meters
        max_total_distance = direct_distance * max_detour_factor
        
        journeys = []
        
        # 1. Direct journey (no stops)
        direct_journey = self._create_direct_journey(
            start_station, end_station, start_coord, end_coord, hour, day_of_week
        )
        journeys.append(direct_journey)
        
        # 2. Single-stop journeys
        single_stop_journeys = self._predict_single_stop_journeys(
            start_station, end_station, start_coord, end_coord, 
            hour, day_of_week, max_total_distance
        )
        journeys.extend(single_stop_journeys)
        
        # 3. Multi-stop journeys (if requested)
        if max_stops > 1:
            multi_stop_journeys = self._predict_multi_stop_journeys(
                start_station, end_station, start_coord, end_coord,
                hour, day_of_week, max_stops, max_total_distance
            )
            journeys.extend(multi_stop_journeys)
        
        # Sort by probability and return top options
        journeys.sort(key=lambda j: j.visit_probability, reverse=True)
        
        return journeys[:5]  # Return top 5 journey options
    
    def _create_direct_journey(self, start_station: str, end_station: str,
                             start_coord: Tuple[float, float], end_coord: Tuple[float, float],
                             hour: int, day_of_week: int) -> MultiStopJourney:
        """Create a direct journey with no POI stops"""
        
        # Get multiple path options
        paths = self.multi_router.get_multiple_paths(
            start_coord[0], start_coord[1], end_coord[0], end_coord[1], max_paths=1
        )
        
        if paths:
            path = paths[0]
            distance = path.distance_m
            duration = path.duration_s / 60  # Convert to minutes
            coordinates = path.coordinates
        else:
            distance = geodesic(start_coord, end_coord).meters
            duration = distance / 250 * 60  # ~15 km/h in minutes
            coordinates = [start_coord, end_coord]
        
        segment = JourneySegment(
            from_lat=start_coord[0],
            from_lon=start_coord[1],
            to_lat=end_coord[0],
            to_lon=end_coord[1],
            distance_m=distance,
            duration_minutes=duration,
            path_coordinates=coordinates,
            segment_type='direct'
        )
        
        # Get population context
        pop_context = self.station_coverage[start_station].get('population', {})
        
        journey = MultiStopJourney(
            journey_id=f"direct_{start_station}_{end_station}_{hour}",
            origin_station=start_station,
            destination_station=end_station,
            segments=[segment],
            total_distance_m=distance,
            total_duration_minutes=duration,
            visit_probability=0.8,  # Direct routes are common
            journey_type='direct',
            time_of_day=hour,
            day_of_week=day_of_week,
            population_context=pop_context
        )
        
        return journey
    
    def _predict_single_stop_journeys(self, start_station: str, end_station: str,
                                    start_coord: Tuple[float, float], end_coord: Tuple[float, float],
                                    hour: int, day_of_week: int, max_distance: float) -> List[MultiStopJourney]:
        """Predict journeys with single POI stops"""
        
        journeys = []
        
        # Get POIs along the route
        candidate_pois = self._get_pois_along_route(start_coord, end_coord, buffer_m=800)
        
        # Filter and score POIs
        scored_pois = []
        for poi in candidate_pois:
            # Calculate total journey distance with this POI
            dist_to_poi = geodesic(start_coord, (poi.lat, poi.lon)).meters
            dist_from_poi = geodesic((poi.lat, poi.lon), end_coord).meters
            total_distance = dist_to_poi + dist_from_poi
            
            if total_distance <= max_distance:
                # Predict visit probability
                if self.poi_visit_model:
                    features = self._create_poi_features(poi, hour, day_of_week, {})
                    visit_prob = self.poi_visit_model.predict([features])[0]
                else:
                    visit_prob = self._simulate_visit_probability(poi, hour, day_of_week)
                
                scored_pois.append((poi, visit_prob, total_distance))
        
        # Create journeys for top POIs
        scored_pois.sort(key=lambda x: x[1], reverse=True)  # Sort by visit probability
        
        for poi, visit_prob, total_distance in scored_pois[:3]:  # Top 3 POIs
            journey = self._create_single_stop_journey(
                start_station, end_station, start_coord, end_coord,
                poi, hour, day_of_week, visit_prob
            )
            if journey:
                journeys.append(journey)
        
        return journeys
    
    def _create_single_stop_journey(self, start_station: str, end_station: str,
                                  start_coord: Tuple[float, float], end_coord: Tuple[float, float],
                                  poi: POI, hour: int, day_of_week: int, visit_prob: float) -> MultiStopJourney:
        """Create a journey with a single POI stop"""
        
        poi_coord = (poi.lat, poi.lon)
        
        # Get routing for both segments
        paths_to_poi = self.multi_router.get_multiple_paths(
            start_coord[0], start_coord[1], poi_coord[0], poi_coord[1], max_paths=1
        )
        
        paths_from_poi = self.multi_router.get_multiple_paths(
            poi_coord[0], poi_coord[1], end_coord[0], end_coord[1], max_paths=1
        )
        
        # Create segments
        segments = []
        total_distance = 0
        total_duration = 0
        
        # Segment 1: Start to POI
        if paths_to_poi:
            path1 = paths_to_poi[0]
            segment1 = JourneySegment(
                from_lat=start_coord[0],
                from_lon=start_coord[1],
                to_lat=poi_coord[0],
                to_lon=poi_coord[1],
                distance_m=path1.distance_m,
                duration_minutes=path1.duration_s / 60,
                path_coordinates=path1.coordinates,
                segment_type='to_poi'
            )
            total_distance += path1.distance_m
            total_duration += path1.duration_s / 60
        else:
            dist = geodesic(start_coord, poi_coord).meters
            dur = dist / 250 * 60
            segment1 = JourneySegment(
                from_lat=start_coord[0],
                from_lon=start_coord[1],
                to_lat=poi_coord[0],
                to_lon=poi_coord[1],
                distance_m=dist,
                duration_minutes=dur,
                path_coordinates=[start_coord, poi_coord],
                segment_type='to_poi'
            )
            total_distance += dist
            total_duration += dur
        
        segments.append(segment1)
        
        # POI visit duration
        poi_visit_duration = self._estimate_poi_visit_duration(poi, hour, day_of_week)
        total_duration += poi_visit_duration
        
        # Segment 2: POI to End
        if paths_from_poi:
            path2 = paths_from_poi[0]
            segment2 = JourneySegment(
                from_lat=poi_coord[0],
                from_lon=poi_coord[1],
                to_lat=end_coord[0],
                to_lon=end_coord[1],
                poi=poi,
                distance_m=path2.distance_m,
                duration_minutes=path2.duration_s / 60,
                path_coordinates=path2.coordinates,
                segment_type='from_poi'
            )
            total_distance += path2.distance_m
            total_duration += path2.duration_s / 60
        else:
            dist = geodesic(poi_coord, end_coord).meters
            dur = dist / 250 * 60
            segment2 = JourneySegment(
                from_lat=poi_coord[0],
                from_lon=poi_coord[1],
                to_lat=end_coord[0],
                to_lon=end_coord[1],
                poi=poi,
                distance_m=dist,
                duration_minutes=dur,
                path_coordinates=[poi_coord, end_coord],
                segment_type='from_poi'
            )
            total_distance += dist
            total_duration += dur
        
        segments.append(segment2)
        
        # Determine journey type
        journey_type = self._classify_journey_type(poi, hour, day_of_week)
        
        # Get population context
        pop_context = self.station_coverage[start_station].get('population', {})
        
        journey = MultiStopJourney(
            journey_id=f"single_{start_station}_{poi.poi_id}_{end_station}_{hour}",
            origin_station=start_station,
            destination_station=end_station,
            segments=segments,
            total_distance_m=total_distance,
            total_duration_minutes=total_duration,
            visit_probability=visit_prob * 0.7,  # Slightly lower than direct
            journey_type=journey_type,
            time_of_day=hour,
            day_of_week=day_of_week,
            population_context=pop_context
        )
        
        return journey
    
    def _predict_multi_stop_journeys(self, start_station: str, end_station: str,
                                   start_coord: Tuple[float, float], end_coord: Tuple[float, float],
                                   hour: int, day_of_week: int, max_stops: int, max_distance: float) -> List[MultiStopJourney]:
        """Predict journeys with multiple POI stops"""
        
        journeys = []
        
        # For multi-stop, we'll use a simplified approach
        # In practice, this would use more sophisticated route optimization
        
        # Get candidate POIs
        candidate_pois = self._get_pois_along_route(start_coord, end_coord, buffer_m=1000)
        
        # Create combinations of 2 POIs for now
        if len(candidate_pois) >= 2:
            # Sort POIs by visit probability
            scored_pois = []
            for poi in candidate_pois:
                if self.poi_visit_model:
                    features = self._create_poi_features(poi, hour, day_of_week, {})
                    visit_prob = self.poi_visit_model.predict([features])[0]
                else:
                    visit_prob = self._simulate_visit_probability(poi, hour, day_of_week)
                scored_pois.append((poi, visit_prob))
            
            scored_pois.sort(key=lambda x: x[1], reverse=True)
            
            # Try combinations of top POIs
            for i in range(min(3, len(scored_pois))):
                for j in range(i+1, min(5, len(scored_pois))):
                    poi1, prob1 = scored_pois[i]
                    poi2, prob2 = scored_pois[j]
                    
                    # Check if this route is reasonable
                    journey = self._create_multi_stop_journey(
                        start_station, end_station, start_coord, end_coord,
                        [poi1, poi2], hour, day_of_week, max_distance
                    )
                    
                    if journey:
                        journeys.append(journey)
        
        return journeys[:2]  # Return top 2 multi-stop journeys
    
    def _create_multi_stop_journey(self, start_station: str, end_station: str,
                                 start_coord: Tuple[float, float], end_coord: Tuple[float, float],
                                 pois: List[POI], hour: int, day_of_week: int, max_distance: float) -> MultiStopJourney:
        """Create a journey with multiple POI stops"""
        
        # Simple implementation: visit POIs in order of distance from start
        poi_coords = [(poi.lat, poi.lon) for poi in pois]
        
        # Calculate total distance to check feasibility
        total_distance = 0
        coords = [start_coord] + poi_coords + [end_coord]
        
        for i in range(len(coords) - 1):
            total_distance += geodesic(coords[i], coords[i+1]).meters
        
        if total_distance > max_distance:
            return None
        
        # Create segments
        segments = []
        total_duration = 0
        
        current_coord = start_coord
        
        # Segments to each POI
        for i, poi in enumerate(pois):
            poi_coord = (poi.lat, poi.lon)
            
            # Get path
            paths = self.multi_router.get_multiple_paths(
                current_coord[0], current_coord[1], poi_coord[0], poi_coord[1], max_paths=1
            )
            
            if paths:
                path = paths[0]
                segment = JourneySegment(
                    from_lat=current_coord[0],
                    from_lon=current_coord[1],
                    to_lat=poi_coord[0],
                    to_lon=poi_coord[1],
                    poi=poi if i > 0 else None,
                    distance_m=path.distance_m,
                    duration_minutes=path.duration_s / 60,
                    path_coordinates=path.coordinates,
                    segment_type='to_poi'
                )
                total_duration += path.duration_s / 60
            else:
                dist = geodesic(current_coord, poi_coord).meters
                dur = dist / 250 * 60
                segment = JourneySegment(
                    from_lat=current_coord[0],
                    from_lon=current_coord[1],
                    to_lat=poi_coord[0],
                    to_lon=poi_coord[1],
                    poi=poi if i > 0 else None,
                    distance_m=dist,
                    duration_minutes=dur,
                    path_coordinates=[current_coord, poi_coord],
                    segment_type='to_poi'
                )
                total_duration += dur
            
            segments.append(segment)
            
            # Add POI visit time
            poi_visit_duration = self._estimate_poi_visit_duration(poi, hour, day_of_week)
            total_duration += poi_visit_duration
            
            current_coord = poi_coord
        
        # Final segment to destination
        paths = self.multi_router.get_multiple_paths(
            current_coord[0], current_coord[1], end_coord[0], end_coord[1], max_paths=1
        )
        
        if paths:
            path = paths[0]
            final_segment = JourneySegment(
                from_lat=current_coord[0],
                from_lon=current_coord[1],
                to_lat=end_coord[0],
                to_lon=end_coord[1],
                distance_m=path.distance_m,
                duration_minutes=path.duration_s / 60,
                path_coordinates=path.coordinates,
                segment_type='to_destination'
            )
            total_duration += path.duration_s / 60
        else:
            dist = geodesic(current_coord, end_coord).meters
            dur = dist / 250 * 60
            final_segment = JourneySegment(
                from_lat=current_coord[0],
                from_lon=current_coord[1],
                to_lat=end_coord[0],
                to_lon=end_coord[1],
                distance_m=dist,
                duration_minutes=dur,
                path_coordinates=[current_coord, end_coord],
                segment_type='to_destination'
            )
            total_duration += dur
        
        segments.append(final_segment)
        
        # Calculate combined visit probability
        visit_probs = []
        for poi in pois:
            if self.poi_visit_model:
                features = self._create_poi_features(poi, hour, day_of_week, {})
                prob = self.poi_visit_model.predict([features])[0]
            else:
                prob = self._simulate_visit_probability(poi, hour, day_of_week)
            visit_probs.append(prob)
        
        combined_prob = np.mean(visit_probs) * 0.5  # Multi-stop less likely
        
        # Determine journey type
        journey_type = 'errands'  # Multi-stop usually for errands
        
        # Get population context
        pop_context = self.station_coverage[start_station].get('population', {})
        
        journey = MultiStopJourney(
            journey_id=f"multi_{start_station}_{len(pois)}stops_{end_station}_{hour}",
            origin_station=start_station,
            destination_station=end_station,
            segments=segments,
            total_distance_m=total_distance,
            total_duration_minutes=total_duration,
            visit_probability=combined_prob,
            journey_type=journey_type,
            time_of_day=hour,
            day_of_week=day_of_week,
            population_context=pop_context
        )
        
        return journey
    
    def _estimate_poi_visit_duration(self, poi: POI, hour: int, day_of_week: int) -> float:
        """Estimate how long someone spends at a POI (in minutes)"""
        base_durations = {
            'restaurants': 45,
            'cafes': 20,
            'shops': 15,
            'supermarkets': 25,
            'banks': 10,
            'atms': 3,
            'schools': 30,
            'parks': 30,
            'hospitals': 60,
            'pharmacies': 8,
            'bus_stops': 2,
            'train_stations': 5
        }
        
        base_duration = base_durations.get(poi.poi_type, 15)
        
        # Adjust by time of day
        if poi.poi_type in ['restaurants'] and hour in [12, 13, 19, 20]:
            base_duration *= 1.3  # Longer meals during meal times
        elif poi.poi_type in ['cafes'] and hour in [8, 9, 15, 16]:
            base_duration *= 1.2  # Longer coffee breaks
        
        return base_duration
    
    def _classify_journey_type(self, poi: POI, hour: int, day_of_week: int) -> str:
        """Classify journey type based on POI and time"""
        if poi.poi_type in ['restaurants', 'cafes']:
            if hour in [12, 13]:
                return 'lunch'
            elif hour in [19, 20, 21]:
                return 'dining'
            else:
                return 'leisure'
        elif poi.poi_type in ['banks', 'atms', 'shops', 'supermarkets']:
            return 'errands'
        elif poi.poi_type in ['schools', 'offices']:
            return 'work'
        elif poi.poi_type in ['parks', 'museums']:
            return 'leisure'
        elif poi.poi_type in ['hospitals', 'pharmacies']:
            return 'health'
        else:
            return 'other'
    
    def get_journey_summary(self) -> Dict:
        """Get summary of the journey prediction system"""
        return {
            'poi_count': len(self.poi_database),
            'station_coverage': len(self.station_coverage),
            'poi_graph_nodes': len(self.poi_graph.nodes) if self.poi_graph else 0,
            'poi_graph_edges': len(self.poi_graph.edges) if self.poi_graph else 0,
            'models_trained': {
                'poi_visit_model': self.poi_visit_model is not None,
                'journey_type_model': self.journey_type_model is not None,
                'duration_model': self.duration_model is not None
            }
        }

if __name__ == "__main__":
    # Demo usage
    print("🚴‍♂️ Multi-Stop Journey Predictor Demo")
    
    # This would be integrated with your existing system
    print("This predictor creates realistic journey patterns with POI stops!")
    print("Features:")
    print("- Predicts intermediate POI visits (banks, restaurants, etc.)")
    print("- Multiple path options for each journey segment")
    print("- Time-aware visit probability")
    print("- Population-context aware routing")
    print("- Journey type classification")
