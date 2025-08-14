"""
Optimized Switzerland OSM Feature to Station Mapper
Fast extraction of OSM features near train stations for ML training.
"""

import osmium
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from geopy.distance import geodesic
from dataclasses import dataclass
import logging
import os
import json
from collections import defaultdict
import time
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OSMFeature:
    """Represents an OSM feature with its properties"""
    feature_type: str
    feature_value: str
    lat: float
    lon: float
    name: str = ""
    
    def distance_to(self, lat: float, lon: float) -> float:
        """Calculate distance to coordinates in meters"""
        return geodesic((self.lat, self.lon), (lat, lon)).meters

class OptimizedOSMHandler(osmium.SimpleHandler):
    """Optimized handler to extract features from OSM PBF file"""
    
    def __init__(self, feature_mappings: Dict, stations_bbox: Tuple[float, float, float, float], max_distance_km: float = 2.0):
        osmium.SimpleHandler.__init__(self)
        self.features = []
        self.feature_mappings = feature_mappings
        self.stations_bbox = stations_bbox  # (min_lat, min_lon, max_lat, max_lon)
        self.max_distance_km = max_distance_km
        self.processed_count = 0
        self.relevant_count = 0
        self.start_time = time.time()
        
        # Expand bbox by max distance
        lat_buffer = max_distance_km / 111.0  # ~111km per degree latitude
        avg_lat = (stations_bbox[0] + stations_bbox[2]) / 2
        lon_buffer = max_distance_km / (111.0 * np.cos(np.radians(avg_lat)))
        
        self.bbox = (
            stations_bbox[0] - lat_buffer,  # min_lat
            stations_bbox[1] - lon_buffer,  # min_lon
            stations_bbox[2] + lat_buffer,  # max_lat
            stations_bbox[3] + lon_buffer   # max_lon
        )
        
        logger.info(f"Filtering to bbox: {self.bbox}")
        
    def node(self, n):
        """Process OSM nodes"""
        self.processed_count += 1
        
        if self.processed_count % 500000 == 0:
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed
            logger.info(f"Processed {self.processed_count:,} nodes in {elapsed:.1f}s ({rate:.0f} nodes/s), found {self.relevant_count:,} relevant features")
        
        # Quick bbox filter
        lat, lon = n.location.lat, n.location.lon
        if not (self.bbox[0] <= lat <= self.bbox[2] and self.bbox[1] <= lon <= self.bbox[3]):
            return
        
        # Extract tags
        tags = dict(n.tags)
        
        # Categorize feature
        osm_feature = self._categorize_feature(tags, lat, lon)
        if osm_feature:
            self.features.append(osm_feature)
            self.relevant_count += 1
    
    def _categorize_feature(self, tags: Dict, lat: float, lon: float) -> Optional[OSMFeature]:
        """Categorize a feature based on its tags"""
        
        # Priority order for feature categorization
        priority_tags = ['amenity', 'shop', 'tourism', 'railway', 'public_transport', 'leisure', 'office']
        
        for main_tag in priority_tags:
            if main_tag in tags and main_tag in self.feature_mappings:
                tag_value = tags[main_tag]
                tag_mappings = self.feature_mappings[main_tag]
                
                # Find specific mapping or use catch-all
                feature_category = None
                if tag_value in tag_mappings:
                    feature_category = tag_mappings[tag_value]
                elif "*" in tag_mappings:
                    feature_category = tag_mappings["*"]
                
                if feature_category:
                    name = tags.get('name', tags.get('ref', ''))
                    
                    return OSMFeature(
                        feature_type=feature_category,
                        feature_value=tag_value,
                        lat=lat,
                        lon=lon,
                        name=name
                    )
        
        return None

class OptimizedSwissStationMapper:
    """Fast OSM feature extraction for Swiss train stations"""
    
    def __init__(self, pbf_path: str = "data/switzerland-latest.osm.pbf", 
                 stations_path: str = "data/unique_stations.csv",
                 output_dir: str = "cache/ml_ready"):
        self.pbf_path = pbf_path
        self.stations_path = stations_path
        self.output_dir = output_dir
        self.features = []
        self.stations = None
        self._ensure_output_dir()
        
        # Streamlined feature mappings focusing on bike-relevant features
        self.feature_mappings = {
            "amenity": {
                "restaurant": "restaurants",
                "cafe": "cafes", 
                "fast_food": "restaurants",
                "bar": "bars",
                "pub": "bars",
                "bank": "banks",
                "atm": "atms",
                "pharmacy": "pharmacies",
                "hospital": "hospitals",
                "clinic": "clinics",
                "dentist": "dentists",
                "school": "schools",
                "university": "universities",
                "library": "libraries",
                "bus_station": "bus_stops",
                "parking": "parking",
                "fuel": "fuel",
                "police": "police",
                "post_office": "postal_services",
                "cinema": "cinemas",
                "theatre": "theatres",
                "place_of_worship": "religious",
                "gym": "gyms",
                "fitness_centre": "gyms",
                "bicycle_parking": "bike_parking",
                "bicycle_rental": "bike_rental",
                "toilets": "public_toilets"
            },
            "shop": {
                "supermarket": "supermarkets",
                "convenience": "convenience_stores",
                "clothes": "clothing_stores",
                "bakery": "bakeries",
                "bicycle": "bike_shops",
                "electronics": "electronics_stores",
                "books": "bookstores",
                "hairdresser": "personal_services",
                "mall": "shopping_centers",
                "department_store": "shopping_centers",
                "*": "general_shops"
            },
            "tourism": {
                "hotel": "hotels",
                "guest_house": "hotels",
                "hostel": "hotels",
                "museum": "museums",
                "attraction": "tourist_attractions",
                "information": "tourist_info",
                "viewpoint": "viewpoints",
                "*": "tourism_general"
            },
            "leisure": {
                "park": "parks",
                "playground": "playgrounds",
                "sports_centre": "sports_centres",
                "stadium": "stadiums",
                "swimming_pool": "swimming_pools",
                "fitness_centre": "fitness_centres",
                "*": "leisure_general"
            },
            "highway": {
                "bus_stop": "bus_stops",
                "traffic_signals": "traffic_lights",
                "*": None
            },
            "railway": {
                "station": "train_stations",
                "halt": "train_stops",
                "tram_stop": "tram_stops",
                "*": "railway_general"
            },
            "public_transport": {
                "stop_position": "transit_stops",
                "platform": "transit_platforms",
                "station": "transit_stations",
                "*": "public_transport_general"
            },
            "office": {
                "company": "offices",
                "government": "government_offices",
                "*": "general_offices"
            }
        }
        
    def _ensure_output_dir(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def load_stations(self):
        """Load station data from CSV"""
        logger.info(f"Loading stations from {self.stations_path}")
        self.stations = pd.read_csv(self.stations_path)
        logger.info(f"Loaded {len(self.stations)} unique stations")
        return self.stations
    
    def get_stations_bbox(self) -> Tuple[float, float, float, float]:
        """Get bounding box of all stations"""
        if self.stations is None:
            self.load_stations()
        
        bbox = (
            self.stations['lat'].min(),
            self.stations['lon'].min(),
            self.stations['lat'].max(),
            self.stations['lon'].max()
        )
        
        logger.info(f"Stations bounding box: {bbox}")
        return bbox
    
    def parse_pbf_features_fast(self) -> List[OSMFeature]:
        """Fast parsing of PBF file with spatial filtering"""
        
        # Try loading from cache first
        if self._load_features_cache():
            return self.features
        
        logger.info(f"Fast parsing of PBF file: {self.pbf_path}")
        
        # Get stations bounding box
        stations_bbox = self.get_stations_bbox()
        
        # Create optimized handler
        handler = OptimizedOSMHandler(self.feature_mappings, stations_bbox, max_distance_km=2.0)
        
        # Apply handler to PBF file
        start_time = time.time()
        logger.info("Starting PBF processing... this will take a few minutes")
        
        handler.apply_file(self.pbf_path)
        
        elapsed = time.time() - start_time
        logger.info(f"Completed processing in {elapsed:.1f} seconds")
        logger.info(f"Processed {handler.processed_count:,} total nodes")
        logger.info(f"Extracted {len(handler.features):,} relevant features")
        
        self.features = handler.features
        
        # Save features to cache for faster reuse
        self._save_features_cache()
        
        return self.features
    
    def _save_features_cache(self):
        """Save extracted features to cache"""
        if not self.features:
            return
            
        cache_file = os.path.join(self.output_dir, "switzerland_osm_features.json")
        logger.info(f"Saving {len(self.features)} features to cache: {cache_file}")
        
        # Convert features to serializable format
        features_data = []
        for feature in self.features:
            features_data.append({
                'feature_type': feature.feature_type,
                'feature_value': feature.feature_value,
                'lat': feature.lat,
                'lon': feature.lon,
                'name': feature.name
            })
        
        with open(cache_file, 'w') as f:
            json.dump(features_data, f, indent=2)
        logger.info(f"Features cache saved successfully")
    
    def _load_features_cache(self) -> bool:
        """Load features from cache if available"""
        cache_file = os.path.join(self.output_dir, "switzerland_osm_features.json")
        
        if not os.path.exists(cache_file):
            return False
            
        logger.info(f"Loading features from cache: {cache_file}")
        
        with open(cache_file, 'r') as f:
            features_data = json.load(f)
        
        self.features = [
            OSMFeature(
                feature_type=f['feature_type'],
                feature_value=f['feature_value'],
                lat=f['lat'],
                lon=f['lon'],
                name=f.get('name', '')
            ) for f in features_data
        ]
        
        logger.info(f"Loaded {len(self.features)} features from cache")
        return True
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Fast haversine distance calculation in meters"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def _build_spatial_grid(self, cell_size_deg: float = 0.01) -> Dict[Tuple[int, int], List[int]]:
        """Build spatial grid index for fast lookups"""
        logger.info("Building spatial grid index...")
        grid = defaultdict(list)
        
        for idx, feature in enumerate(self.features):
            grid_x = int(feature.lat / cell_size_deg)
            grid_y = int(feature.lon / cell_size_deg)
            grid[(grid_x, grid_y)].append(idx)
        
        logger.info(f"Built grid with {len(grid)} cells, avg {len(self.features)/len(grid):.1f} features per cell")
        return dict(grid)

    def _get_nearby_grid_cells(self, lat: float, lon: float, radius_meters: float, cell_size_deg: float = 0.01) -> List[Tuple[int, int]]:
        """Get grid cells that might contain features within radius"""
        # Convert radius to approximate degrees
        lat_buffer = radius_meters / 111000  # ~111km per degree latitude
        lon_buffer = radius_meters / (111000 * math.cos(math.radians(lat)))  # Adjust for longitude
        
        center_x = int(lat / cell_size_deg)
        center_y = int(lon / cell_size_deg)
        
        x_range = int(lat_buffer / cell_size_deg) + 1
        y_range = int(lon_buffer / cell_size_deg) + 1
        
        cells = []
        for x in range(center_x - x_range, center_x + x_range + 1):
            for y in range(center_y - y_range, center_y + y_range + 1):
                cells.append((x, y))
        
        return cells

    def calculate_station_features_optimized(self, radius_meters: int = 1000) -> pd.DataFrame:
        """OPTIMIZED: Calculate features within radius using spatial grid indexing"""
        if self.stations is None:
            self.load_stations()
        
        if not self.features:
            logger.warning("No features loaded. Please run parse_pbf_features_fast() first.")
            return pd.DataFrame()
        
        logger.info(f"🚀 OPTIMIZED: Calculating features within {radius_meters}m of {len(self.stations)} stations")
        start_time = time.time()
        
        # Build spatial grid index
        grid = self._build_spatial_grid()
        
        # Get all unique feature types for consistent columns
        all_feature_types = sorted(set(f.feature_type for f in self.features if f.feature_type))
        
        # Initialize result storage
        station_features = []
        
        logger.info("Starting spatial queries...")
        for idx, station in self.stations.iterrows():
            station_id = station['station_id']
            station_lat = station['lat']
            station_lon = station['lon']
            
            # Get nearby grid cells
            nearby_cells = self._get_nearby_grid_cells(station_lat, station_lon, radius_meters)
            
            # Count features by type within radius
            feature_counts = defaultdict(int)
            nearest_distances = defaultdict(lambda: radius_meters)
            
            # Check features in nearby grid cells only
            checked_features = set()
            for cell in nearby_cells:
                if cell in grid:
                    for feature_idx in grid[cell]:
                        if feature_idx in checked_features:
                            continue
                        checked_features.add(feature_idx)
                        
                        feature = self.features[feature_idx]
                        
                        # Calculate distance using fast haversine
                        distance = self._haversine_distance(station_lat, station_lon, feature.lat, feature.lon)
                        
                        if distance <= radius_meters:
                            feature_type = feature.feature_type
                            feature_counts[feature_type] += 1
                            
                            # Track nearest distance of each type
                            if distance < nearest_distances[feature_type]:
                                nearest_distances[feature_type] = distance
            
            # Create feature vector for this station
            station_feature_dict = {
                'station_id': station_id,
                'lat': station_lat,
                'lon': station_lon,
                'coords': station['coords']
            }
            
            # Add counts and distances for each feature type
            for feature_type in all_feature_types:
                station_feature_dict[f'{feature_type}_count'] = feature_counts[feature_type]
                station_feature_dict[f'{feature_type}_nearest_distance'] = nearest_distances[feature_type]
            
            # Add total feature count
            station_feature_dict['total_features'] = sum(feature_counts.values())
            
            station_features.append(station_feature_dict)
            
            # Log progress more frequently for feedback
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta = (len(self.stations) - idx - 1) / rate if rate > 0 else 0
                logger.info(f"Processed {idx + 1}/{len(self.stations)} stations ({rate:.1f} stations/sec, ETA: {eta:.1f}s)")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Spatial processing completed in {elapsed:.1f} seconds ({len(self.stations)/elapsed:.1f} stations/sec)")
        
        return pd.DataFrame(station_features)

    def calculate_station_features(self, radius_meters: int = 1000) -> pd.DataFrame:
        """Wrapper that calls optimized version"""
        return self.calculate_station_features_optimized(radius_meters)
    
    def create_feature_summary(self) -> Dict:
        """Create a summary of extracted features"""
        if not self.features:
            return {}
        
        feature_summary = defaultdict(int)
        for feature in self.features:
            if feature.feature_type:
                feature_summary[feature.feature_type] += 1
        
        return dict(feature_summary)
    
    def save_results(self, station_features_df: pd.DataFrame, radius_meters: int = 1000):
        """Save results to files"""
        
        # Save station features dataset
        features_file = os.path.join(self.output_dir, f'switzerland_station_features_{radius_meters}m.csv')
        station_features_df.to_csv(features_file, index=False)
        logger.info(f"Saved station features to {features_file}")
        
        # Save feature summary
        summary = self.create_feature_summary()
        summary_file = os.path.join(self.output_dir, 'switzerland_osm_feature_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved feature summary to {summary_file}")
        
        return features_file, summary_file

def main():
    """Main execution function"""
    logger.info("Starting Optimized Switzerland OSM Station Mapping...")
    
    # Initialize mapper
    mapper = OptimizedSwissStationMapper()
    
    # Load stations
    stations = mapper.load_stations()
    
    # Fast parse OSM features from PBF
    features = mapper.parse_pbf_features_fast()
    
    # Calculate features for different radii
    radii = [500, 1000, 1500]  # Different search radii in meters
    
    for radius in radii:
        logger.info(f"\n--- Processing radius: {radius}m ---")
        
        # Calculate station features
        station_features_df = mapper.calculate_station_features(radius)
        
        if not station_features_df.empty:
            # Save results
            features_file, summary_file = mapper.save_results(station_features_df, radius)
            
            # Print summary statistics
            print(f"\n=== SUMMARY FOR {radius}m RADIUS ===")
            print(f"Total stations: {len(station_features_df)}")
            print(f"Total OSM features extracted: {len(features)}")
            print(f"Average features per station: {station_features_df['total_features'].mean():.2f}")
            print(f"Max features at station: {station_features_df['total_features'].max()}")
            print(f"Min features at station: {station_features_df['total_features'].min()}")
            
            # Show top feature types
            feature_summary = mapper.create_feature_summary()
            print(f"\nTop 10 feature types by count:")
            for feature_type, count in sorted(feature_summary.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {feature_type}: {count:,}")
    
    logger.info("Switzerland OSM Station Mapping completed successfully!")

if __name__ == "__main__":
    main()
