"""
OSM GeoJSON Parser for ML Algorithm
Parses GeoJSON files and formats data for compatibility with the ML prediction system
"""

import json
import gzip
import os
import time
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from geopy.distance import geodesic
from dataclasses import dataclass
import lzma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OSMGeoJSONFeature:
    """Representation of a feature from OSM GeoJSON"""
    feature_type: str  # Main category (amenity, shop, etc)
    feature_value: str  # Specific value (restaurant, cafe, etc)
    lat: float
    lon: float
    name: str = ""
    tags: Dict = None
    geometry_type: str = "Point"  # Point, LineString, Polygon, etc.
    
    def distance_to(self, lat: float, lon: float) -> float:
        """Calculate distance to coordinates in meters"""
        return geodesic((self.lat, self.lon), (lat, lon)).meters

class OSMGeoJSONParser:
    """Parse OSM GeoJSON files for ML feature extraction"""
    
    def __init__(self, cache_dir: str = "cache/geojson_cache"):
        self.cache_dir = cache_dir
        self.features = {}  # Cache for extracted features
        self._ensure_cache_dir()
        
        # Define OSM feature mappings
        self.feature_mappings = {
            "amenity": {
                "restaurant": "restaurants",
                "cafe": "cafes", 
                "fast_food": "restaurants",
                "bank": "banks",
                "atm": "atms",
                "pharmacy": "pharmacies",
                "hospital": "hospitals",
                "school": "schools",
                "university": "universities",
                "bus_station": "bus_stops",
                "parking": "parking",
                "fuel": "fuel",
            },
            "shop": {
                "supermarket": "supermarkets",
                "convenience": "shops",
                "clothes": "shops",
                "bakery": "shops",
                "butcher": "shops",
                "*": "shops"  # Catch-all for other shop types
            },
            "tourism": {
                "hotel": "hotels",
                "guest_house": "hotels",
                "hostel": "hotels",
                "museum": "museums",
                "attraction": "tourism",
                "*": "tourism"  # Catch-all for other tourism types
            },
            "leisure": {
                "park": "parks",
                "garden": "parks",
                "playground": "parks",
                "sports_centre": "sports_centres",
                "stadium": "sports_centres",
                "*": "leisure"  # Catch-all for other leisure types
            },
            "highway": {
                "bus_stop": "bus_stops",
                "pedestrian": "pedestrian_areas",
                "footway": "pedestrian_areas",
                "path": "pedestrian_areas",
                "*": None  # Ignore most highway features
            },
            "railway": {
                "station": "train_stations",
                "stop": "train_stations",
                "tram_stop": "tram_stops",
                "*": "railway"  # Catch-all for other railway types
            },
            "building": {
                "residential": "residential",
                "commercial": "commercial",
                "retail": "commercial",
                "office": "office",
                "industrial": "industrial",
                "*": "buildings"  # Catch-all for other building types
            },
            "landuse": {
                "residential": "residential",
                "commercial": "commercial",
                "retail": "commercial",
                "industrial": "industrial",
                "park": "parks",
                "*": "landuse"  # Catch-all for other landuse types
            },
        }
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_geojson(self, file_path: str) -> List[OSMGeoJSONFeature]:
        """
        Load GeoJSON file and extract features
        
        Args:
            file_path: Path to GeoJSON file (can be .geojson, .geojson.gz, or .geojson.xz)
            
        Returns:
            List of parsed OSM features
        """
        cache_key = os.path.basename(file_path).replace('.', '_')
        cache_file = os.path.join(self.cache_dir, f"{cache_key}_parsed.json.gz")
        
        # Check cache first
        if os.path.exists(cache_file):
            logger.info(f"Loading cached parsed features from {cache_file}")
            try:
                with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    features = []
                    for item in data:
                        feature = OSMGeoJSONFeature(
                            feature_type=item['feature_type'],
                            feature_value=item['feature_value'],
                            lat=item['lat'],
                            lon=item['lon'],
                            name=item['name'],
                            tags=item['tags'],
                            geometry_type=item['geometry_type']
                        )
                        features.append(feature)
                    logger.info(f"Loaded {len(features)} features from cache")
                    return features
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        logger.info(f"Parsing GeoJSON file: {file_path}")
        start_time = time.time()
        
        # Open file based on extension
        features = []
        if file_path.endswith('.xz'):
            with lzma.open(file_path, 'rt', encoding='utf-8') as f:
                geojson = self._load_json_from_stream(f)
                features = self._extract_features(geojson)
        elif file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                geojson = self._load_json_from_stream(f)
                features = self._extract_features(geojson)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
                features = self._extract_features(geojson)
        
        # Save to cache
        self._save_features_to_cache(features, cache_file)
        
        elapsed = time.time() - start_time
        logger.info(f"Parsed {len(features)} features in {elapsed:.2f} seconds")
        
        return features
    
    def _load_json_from_stream(self, stream):
        """Load JSON from stream with error handling"""
        try:
            return json.load(stream)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            
            # Try to be more forgiving
            logger.info("Trying to parse with more lenient approach...")
            stream.seek(0)
            content = stream.read()
            try:
                # Try to fix common JSON issues
                content = content.replace('nan', 'null')
                content = content.replace('NaN', 'null')
                content = content.replace('Infinity', 'null')
                content = content.replace('-Infinity', 'null')
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error("Failed with lenient parsing as well")
                return {"type": "FeatureCollection", "features": []}
    
    def _extract_features(self, geojson: Dict) -> List[OSMGeoJSONFeature]:
        """
        Extract relevant features from GeoJSON
        
        Args:
            geojson: Parsed GeoJSON data
            
        Returns:
            List of OSM features
        """
        features = []
        
        if 'type' not in geojson or geojson['type'] != 'FeatureCollection':
            logger.warning("Invalid GeoJSON: Not a FeatureCollection")
            return features
        
        if 'features' not in geojson:
            logger.warning("Invalid GeoJSON: No features array")
            return features
        
        # Process each feature
        for feature in geojson['features']:
            try:
                # Skip if no geometry or properties
                if not feature.get('geometry') or not feature.get('properties'):
                    continue
                
                # Get coordinates
                geometry = feature['geometry']
                geometry_type = geometry.get('type', '')
                
                coordinates = geometry.get('coordinates', [])
                lat, lon = None, None
                
                if geometry_type == 'Point':
                    # Point coordinates are [lon, lat]
                    if len(coordinates) >= 2:
                        lon, lat = coordinates[0], coordinates[1]
                elif geometry_type == 'LineString' or geometry_type == 'MultiPoint':
                    # For lines or multi-points, use centroid
                    if coordinates:
                        points = coordinates
                        lon = sum(p[0] for p in points) / len(points)
                        lat = sum(p[1] for p in points) / len(points)
                elif geometry_type == 'Polygon' or geometry_type == 'MultiLineString':
                    # For polygons, use centroid of first ring
                    if coordinates and coordinates[0]:
                        points = coordinates[0]
                        lon = sum(p[0] for p in points) / len(points)
                        lat = sum(p[1] for p in points) / len(points)
                
                # Skip if no valid coordinates
                if lat is None or lon is None:
                    continue
                
                # Get properties and map to our feature types
                properties = feature['properties']
                feature_type = None
                feature_value = None
                
                # Find the main OSM tag (amenity, shop, etc.)
                for key in self.feature_mappings.keys():
                    if key in properties:
                        feature_type = key
                        feature_value = properties[key]
                        break
                
                # Skip if no matching feature type
                if feature_type is None:
                    continue
                
                # Map to our standardized feature categories
                mapped_type = self._map_osm_feature(feature_type, feature_value)
                
                # Skip if feature is to be ignored
                if mapped_type is None:
                    continue
                
                # Create feature object
                osm_feature = OSMGeoJSONFeature(
                    feature_type=mapped_type,
                    feature_value=feature_value,
                    lat=lat,
                    lon=lon,
                    name=properties.get('name', ''),
                    tags=properties,
                    geometry_type=geometry_type
                )
                
                features.append(osm_feature)
                
            except Exception as e:
                # Skip features with errors
                logger.debug(f"Error processing feature: {e}")
                continue
        
        logger.info(f"Extracted {len(features)} features from GeoJSON")
        return features
    
    def _map_osm_feature(self, feature_type: str, feature_value: str) -> Optional[str]:
        """
        Map OSM feature to our standardized categories
        
        Args:
            feature_type: OSM tag key (amenity, shop, etc)
            feature_value: OSM tag value (restaurant, cafe, etc)
            
        Returns:
            Mapped feature category or None if to be ignored
        """
        if feature_type in self.feature_mappings:
            # Check for specific value mapping
            if feature_value in self.feature_mappings[feature_type]:
                return self.feature_mappings[feature_type][feature_value]
            
            # Use catch-all if available
            if '*' in self.feature_mappings[feature_type]:
                return self.feature_mappings[feature_type]['*']
        
        return None  # Ignore features without mapping
    
    def _save_features_to_cache(self, features: List[OSMGeoJSONFeature], cache_file: str):
        """Save parsed features to cache"""
        try:
            data = []
            for feature in features:
                data.append({
                    'feature_type': feature.feature_type,
                    'feature_value': feature.feature_value,
                    'lat': feature.lat,
                    'lon': feature.lon,
                    'name': feature.name,
                    'tags': feature.tags,
                    'geometry_type': feature.geometry_type
                })
            
            with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
                json.dump(data, f)
                
            logger.info(f"Saved {len(features)} features to cache: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def extract_features_around_location(self, features: List[OSMGeoJSONFeature], 
                                      lat: float, lon: float, radius_m: int = 500) -> Dict[str, List[OSMGeoJSONFeature]]:
        """
        Extract features around a given location
        
        Args:
            features: List of all OSM features
            lat: Latitude of center point
            lon: Longitude of center point
            radius_m: Radius in meters
            
        Returns:
            Dictionary of features grouped by type
        """
        nearby_features = {}
        
        for feature in features:
            distance = feature.distance_to(lat, lon)
            
            if distance <= radius_m:
                feature_type = feature.feature_type
                
                if feature_type not in nearby_features:
                    nearby_features[feature_type] = []
                    
                nearby_features[feature_type].append(feature)
        
        logger.info(f"Found {sum(len(v) for v in nearby_features.values())} features within {radius_m}m of ({lat}, {lon})")
        return nearby_features
    
    def compute_feature_metrics(self, features_dict: Dict[str, List[OSMGeoJSONFeature]], 
                             distance_bands: List[int] = [100, 250, 500]) -> Dict[str, float]:
        """
        Compute aggregated metrics for features around a station
        
        Args:
            features_dict: Dictionary of features grouped by type
            distance_bands: Distance bands for counting features
            
        Returns:
            Dictionary of feature metrics
        """
        metrics = {}
        
        for feature_type, feature_list in features_dict.items():
            # Basic counts
            metrics[f'{feature_type}_count'] = len(feature_list)
            
            if feature_list:
                # Distance statistics
                distances = [f.distance_to(f.lat, f.lon) for f in feature_list]
                metrics[f'{feature_type}_min_distance'] = min(distances) if distances else 0
                metrics[f'{feature_type}_avg_distance'] = np.mean(distances) if distances else 0
                metrics[f'{feature_type}_closest_count'] = sum(1 for d in distances if d <= 100)
                
                # Distance band counts
                for band in distance_bands:
                    count = sum(1 for d in distances if d <= band)
                    metrics[f'{feature_type}_within_{band}m'] = count
                
                # Density (features per km²)
                area_km2 = np.pi * (max(distance_bands) / 1000) ** 2
                metrics[f'{feature_type}_density'] = len(feature_list) / area_km2
            else:
                # Zero values for missing features
                metrics[f'{feature_type}_min_distance'] = 0
                metrics[f'{feature_type}_avg_distance'] = 0
                metrics[f'{feature_type}_closest_count'] = 0
                metrics[f'{feature_type}_density'] = 0
                
                for band in distance_bands:
                    metrics[f'{feature_type}_within_{band}m'] = 0
        
        return metrics
    
    def create_feature_dataframe(self, features: List[OSMGeoJSONFeature]) -> pd.DataFrame:
        """
        Create a dataframe with features for ML processing
        
        Args:
            features: List of OSM features
            
        Returns:
            DataFrame with feature counts and metrics
        """
        # Create dataframe with location and feature counts
        feature_types = set(f.feature_type for f in features)
        
        # Find all unique coordinates in the dataset
        coords = set((f.lat, f.lon) for f in features)
        
        # Create rows for dataframe
        rows = []
        for lat, lon in coords:
            # Extract features around this location
            nearby = self.extract_features_around_location(features, lat, lon)
            
            # Compute metrics
            metrics = self.compute_feature_metrics(nearby)
            
            # Create row
            row = {'lat': lat, 'lon': lon}
            row.update(metrics)
            
            rows.append(row)
        
        return pd.DataFrame(rows)

def process_geojson_for_ml(file_path: str, output_dir: str = "cache/ml_ready"):
    """
    Process GeoJSON file and prepare for ML algorithm
    
    Args:
        file_path: Path to GeoJSON file
        output_dir: Directory to save processed features
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Processing {file_path} for ML algorithm")
    
    # Parse GeoJSON
    parser = OSMGeoJSONParser()
    features = parser.load_geojson(file_path)
    
    # Create dataframe
    feature_df = parser.create_feature_dataframe(features)
    
    # Save to file
    output_file = os.path.join(output_dir, "geojson_features.csv")
    feature_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(feature_df)} feature rows to {output_file}")
    
    return feature_df

if __name__ == "__main__":
    # Example usage
    file_path = "data/Bern.osm.geojson.xz"
    df = process_geojson_for_ml(file_path)
    
    print(f"Processed {len(df)} locations")
    print("\nFeature columns:")
    for col in df.columns:
        print(f"- {col}")
    
    print("\nSample data:")
    print(df.head())
