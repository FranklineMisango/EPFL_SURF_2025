"""
Population Feature Extractor for Bike Flow Prediction
Extracts population density and demographic features from population grid data.
Integrates with the OSM feature extractor for comprehensive urban analysis.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import math
import json
import warnings
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
import os

warnings.filterwarnings('ignore')

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

@dataclass
class PopulationFeature:
    """Represents population data for a specific area"""
    lat: float
    lon: float
    population: float
    distance_to_station: float = 0.0
    grid_type: str = 'hectare'  # 'hectare' or 'centroid'

class PopulationFeatureExtractor:
    """Extract population features from population grid data"""
    
    def __init__(self, cache_dir: str = "cache/population_features"):
        self.cache_dir = cache_dir
        self.population_grid = None
        self.population_centroids = None
        self.grid_tree = None  # For fast spatial queries
        self.centroids_tree = None
        self._ensure_cache_dir()
        self._load_population_data()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_population_data(self):
        """Load population data from files"""
        try:
            # Load population grid centroids (CSV)
            logger.info("Loading population grid centroids...")
            self.population_centroids = pd.read_csv('data/pop_grid_centroids.csv')
            
            # Remove zero population entries to reduce memory
            self.population_centroids = self.population_centroids[
                self.population_centroids['pop_count'] > 0
            ].copy()
            
            logger.info(f"Loaded {len(self.population_centroids)} populated grid cells")
            
            # Create spatial index for fast queries
            self._create_spatial_indices()
            
            # Try to load GeoJSON (if not too large)
            try:
                logger.info("Loading population hectare grid...")
                # Try to read the GeoJSON in chunks or sample it
                self._load_geojson_sample()
            except Exception as e:
                logger.warning(f"Could not load full GeoJSON data: {e}")
                self.population_grid = None
                
        except Exception as e:
            logger.error(f"Failed to load population data: {e}")
            self.population_centroids = None
            self.population_grid = None
    
    def _create_spatial_indices(self):
        """Create spatial indices for fast queries"""
        if self.population_centroids is not None:
            # Create KD-tree for fast nearest neighbor queries
            coords = self.population_centroids[['lat', 'lon']].values
            self.centroids_tree = cKDTree(coords)
            logger.info("Created spatial index for population centroids")
    
    def _load_geojson_sample(self):
        """Load a sample of the GeoJSON data for polygon-based queries"""
        try:
            # Try to read the GeoJSON file in chunks or sample
            import json
            
            # For large files, we'll create a simplified version
            sample_file = os.path.join(self.cache_dir, "pop_grid_sample.geojson")
            
            if os.path.exists(sample_file):
                logger.info("Loading cached population grid sample...")
                self.population_grid = gpd.read_file(sample_file)
            else:
                logger.info("Creating population grid sample...")
                # Create a simplified grid based on centroids
                self._create_simplified_grid(sample_file)
                
        except Exception as e:
            logger.warning(f"Failed to load GeoJSON sample: {e}")
            self.population_grid = None
    
    def _create_simplified_grid(self, output_file: str):
        """Create a simplified grid from centroids for basic polygon queries"""
        if self.population_centroids is None:
            return
        
        try:
            # Sample the centroids to reduce size
            sample_size = min(10000, len(self.population_centroids))
            sampled_centroids = self.population_centroids.sample(n=sample_size, random_state=42)
            
            # Create simple square polygons around each centroid
            polygons = []
            for _, row in sampled_centroids.iterrows():
                lat, lon, pop = row['lat'], row['lon'], row['pop_count']
                
                # Create a small square (approximately 100m x 100m)
                offset = 0.0009  # Approximately 100m in degrees
                
                polygon = Polygon([
                    (lon - offset, lat - offset),
                    (lon + offset, lat - offset),
                    (lon + offset, lat + offset),
                    (lon - offset, lat + offset),
                    (lon - offset, lat - offset)
                ])
                
                polygons.append({
                    'geometry': polygon,
                    'population': pop,
                    'lat': lat,
                    'lon': lon
                })
            
            # Create GeoDataFrame
            self.population_grid = gpd.GeoDataFrame(polygons)
            
            # Save for future use
            self.population_grid.to_file(output_file, driver='GeoJSON')
            logger.info(f"Created simplified population grid with {len(polygons)} cells")
            
        except Exception as e:
            logger.warning(f"Failed to create simplified grid: {e}")
            self.population_grid = None
    
    def extract_population_features_around_station(self, lat: float, lon: float, 
                                                 radius_m: int = 500) -> Dict[str, float]:
        """Extract population features around a station within given radius"""
        
        # Check cache first
        cache_key = f"pop_{lat:.6f}_{lon:.6f}_{radius_m}"
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        features = {}
        
        try:
            # Get population data within radius using centroids
            if self.population_centroids is not None and self.centroids_tree is not None:
                pop_features = self._extract_from_centroids(lat, lon, radius_m)
                features.update(pop_features)
            
            # Get polygon-based features if available
            if self.population_grid is not None:
                polygon_features = self._extract_from_polygons(lat, lon, radius_m)
                features.update(polygon_features)
            
            # Cache the result
            self._save_to_cache(cache_key, features)
            
        except Exception as e:
            logger.warning(f"Failed to extract population features: {e}")
            features = self._get_default_features()
        
        return features
    
    def _extract_from_centroids(self, lat: float, lon: float, radius_m: int) -> Dict[str, float]:
        """Extract features using population centroids data"""
        features = {}
        
        try:
            # Convert radius to approximate degrees (rough approximation)
            radius_deg = radius_m / 111000  # 1 degree ≈ 111km
            
            # Find nearby points using KD-tree
            query_point = np.array([[lat, lon]])
            distances, indices = self.centroids_tree.query(
                query_point, 
                k=1000,  # Get up to 1000 nearest points
                distance_upper_bound=radius_deg
            )
            
            # Filter out infinite distances (no more points within radius)
            valid_indices = indices[0][distances[0] != np.inf]
            valid_distances = distances[0][distances[0] != np.inf]
            
            if len(valid_indices) > 0:
                # Get population data for nearby points
                nearby_data = self.population_centroids.iloc[valid_indices].copy()
                
                # Calculate actual distances in meters
                actual_distances = []
                for _, row in nearby_data.iterrows():
                    dist_m = haversine_distance(lat, lon, row['lat'], row['lon'])
                    actual_distances.append(dist_m)
                
                nearby_data['distance_m'] = actual_distances
                
                # Filter by actual radius
                within_radius = nearby_data[nearby_data['distance_m'] <= radius_m]
                
                if len(within_radius) > 0:
                    # Calculate population features
                    features.update(self._calculate_population_metrics(within_radius, radius_m))
                else:
                    features.update(self._get_default_features())
            else:
                features.update(self._get_default_features())
                
        except Exception as e:
            logger.warning(f"Error extracting from centroids: {e}")
            features.update(self._get_default_features())
        
        return features
    
    def _extract_from_polygons(self, lat: float, lon: float, radius_m: int) -> Dict[str, float]:
        """Extract features using polygon data (if available)"""
        features = {}
        
        if self.population_grid is None:
            return features
        
        try:
            # Create query circle
            # Convert radius to approximate degrees
            radius_deg = radius_m / 111000
            
            # Create buffer around point
            query_point = Point(lon, lat)
            query_buffer = query_point.buffer(radius_deg)
            
            # Find intersecting polygons
            intersecting = self.population_grid[
                self.population_grid.geometry.intersects(query_buffer)
            ]
            
            if len(intersecting) > 0:
                # Calculate intersection areas and weighted population
                weighted_population = 0
                total_area = 0
                
                for _, row in intersecting.iterrows():
                    intersection = row.geometry.intersection(query_buffer)
                    intersection_area = intersection.area
                    polygon_area = row.geometry.area
                    
                    if polygon_area > 0:
                        area_ratio = intersection_area / polygon_area
                        weighted_pop = row['population'] * area_ratio
                        weighted_population += weighted_pop
                        total_area += intersection_area
                
                features['polygon_weighted_population'] = weighted_population
                features['polygon_coverage_area'] = total_area
                
        except Exception as e:
            logger.warning(f"Error extracting from polygons: {e}")
        
        return features
    
    def _calculate_population_metrics(self, population_data: pd.DataFrame, radius_m: int) -> Dict[str, float]:
        """Calculate population metrics from nearby population data"""
        metrics = {}
        
        if len(population_data) == 0:
            return self._get_default_features()
        
        # Basic population statistics
        metrics['total_population'] = population_data['pop_count'].sum()
        metrics['average_population_per_cell'] = population_data['pop_count'].mean()
        metrics['max_population_cell'] = population_data['pop_count'].max()
        metrics['min_population_cell'] = population_data['pop_count'].min()
        metrics['population_std'] = population_data['pop_count'].std()
        
        # Population density (people per km²)
        area_km2 = np.pi * (radius_m / 1000) ** 2
        metrics['population_density_km2'] = metrics['total_population'] / area_km2
        
        # Distance-based metrics
        distances = population_data['distance_m']
        metrics['closest_population_distance'] = distances.min()
        metrics['average_population_distance'] = np.average(distances, weights=population_data['pop_count'])
        metrics['farthest_population_distance'] = distances.max()
        
        # Distance bands
        distance_bands = [100, 250, 500, 1000]
        for band in distance_bands:
            if band <= radius_m:
                within_band = population_data[population_data['distance_m'] <= band]
                metrics[f'population_within_{band}m'] = within_band['pop_count'].sum()
                metrics[f'density_within_{band}m'] = (
                    within_band['pop_count'].sum() / (np.pi * (band / 1000) ** 2)
                )
        
        # Population concentration metrics
        if len(population_data) > 1:
            # Gini coefficient for population distribution
            metrics['population_gini'] = self._calculate_gini_coefficient(population_data['pop_count'].values)
            
            # Entropy measure
            pop_proportions = population_data['pop_count'] / population_data['pop_count'].sum()
            metrics['population_entropy'] = -np.sum(pop_proportions * np.log(pop_proportions + 1e-10))
        else:
            metrics['population_gini'] = 0.0
            metrics['population_entropy'] = 0.0
        
        # Spatial distribution
        if len(population_data) > 2:
            # Calculate center of mass
            total_pop = population_data['pop_count'].sum()
            center_lat = np.average(population_data['lat'], weights=population_data['pop_count'])
            center_lon = np.average(population_data['lon'], weights=population_data['pop_count'])
            
            # Distance from station to population center
            center_distance = haversine_distance(
                population_data.iloc[0]['lat'], population_data.iloc[0]['lon'],  # Station location
                center_lat, center_lon
            )
            metrics['population_center_distance'] = center_distance
        else:
            metrics['population_center_distance'] = 0.0
        
        # Grid coverage
        metrics['populated_cells_count'] = len(population_data)
        metrics['empty_cells_ratio'] = 0.0  # We only have populated cells in our data
        
        return metrics
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for population distribution"""
        try:
            values = np.sort(values)
            n = len(values)
            cumsum = np.cumsum(values)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
            return gini
        except:
            return 0.0
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values when no population data is available"""
        return {
            'total_population': 0.0,
            'average_population_per_cell': 0.0,
            'max_population_cell': 0.0,
            'min_population_cell': 0.0,
            'population_std': 0.0,
            'population_density_km2': 0.0,
            'closest_population_distance': 999999.0,
            'average_population_distance': 999999.0,
            'farthest_population_distance': 999999.0,
            'population_within_100m': 0.0,
            'population_within_250m': 0.0,
            'population_within_500m': 0.0,
            'density_within_100m': 0.0,
            'density_within_250m': 0.0,
            'density_within_500m': 0.0,
            'population_gini': 0.0,
            'population_entropy': 0.0,
            'population_center_distance': 0.0,
            'populated_cells_count': 0,
            'empty_cells_ratio': 1.0
        }
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, float]]:
        """Load features from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, features: Dict[str, float]):
        """Save features to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            # Convert numpy types to Python native types for JSON serialization
            serializable_features = {}
            for key, value in features.items():
                if isinstance(value, (np.integer, np.int64)):
                    serializable_features[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    serializable_features[key] = float(value)
                else:
                    serializable_features[key] = value
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_features, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def get_population_summary(self) -> Dict[str, any]:
        """Get summary statistics about the population data"""
        summary = {
            'data_loaded': False,
            'centroids_count': 0,
            'total_population': 0,
            'grid_polygons': 0,
            'spatial_index_ready': False
        }
        
        if self.population_centroids is not None:
            summary['data_loaded'] = True
            summary['centroids_count'] = len(self.population_centroids)
            summary['total_population'] = self.population_centroids['pop_count'].sum()
            summary['spatial_index_ready'] = self.centroids_tree is not None
            
            # Additional statistics
            summary['max_population_cell'] = self.population_centroids['pop_count'].max()
            summary['min_population_cell'] = self.population_centroids['pop_count'].min()
            summary['average_population_cell'] = self.population_centroids['pop_count'].mean()
        
        if self.population_grid is not None:
            summary['grid_polygons'] = len(self.population_grid)
        
        return summary
    
    def analyze_station_population_context(self, station_coords: Dict[str, Tuple[float, float]], 
                                         radius_m: int = 1000) -> pd.DataFrame:
        """Analyze population context for all stations"""
        results = []
        
        logger.info(f"Analyzing population context for {len(station_coords)} stations...")
        
        for i, (station_id, (lat, lon)) in enumerate(station_coords.items()):
            try:
                features = self.extract_population_features_around_station(lat, lon, radius_m)
                
                result = {
                    'station_id': station_id,
                    'lat': lat,
                    'lon': lon,
                    **features
                }
                results.append(result)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(station_coords)} stations")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze station {station_id}: {e}")
                continue
        
        df = pd.DataFrame(results)
        logger.info(f"Completed population analysis for {len(df)} stations")
        
        return df

# Enhanced OSM Feature Extractor with Population Integration
class EnhancedOSMFeatureExtractor:
    """OSM Feature Extractor enhanced with population data"""
    
    def __init__(self, cache_dir: str = "cache/enhanced_features"):
        from osm_feature_extractor import OSMFeatureExtractor
        
        self.osm_extractor = OSMFeatureExtractor(cache_dir=f"{cache_dir}/osm")
        self.population_extractor = PopulationFeatureExtractor(cache_dir=f"{cache_dir}/population")
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def extract_comprehensive_features(self, lat: float, lon: float, 
                                     radius_m: int = 500) -> Dict[str, any]:
        """Extract both OSM and population features"""
        
        # Check cache first
        cache_key = f"comprehensive_{lat:.6f}_{lon:.6f}_{radius_m}"
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        features = {}
        
        try:
            # Extract OSM features
            logger.info(f"Extracting OSM features for ({lat:.6f}, {lon:.6f})")
            osm_features = self.osm_extractor.extract_features_around_station(lat, lon, radius_m)
            osm_metrics = self.osm_extractor.compute_feature_metrics(osm_features)
            
            # Add OSM features with prefix
            for key, value in osm_metrics.items():
                features[f'osm_{key}'] = value
            
            # Extract population features
            logger.info(f"Extracting population features for ({lat:.6f}, {lon:.6f})")
            pop_features = self.population_extractor.extract_population_features_around_station(
                lat, lon, radius_m
            )
            
            # Add population features with prefix
            for key, value in pop_features.items():
                features[f'pop_{key}'] = value
            
            # Calculate interaction features
            interaction_features = self._calculate_interaction_features(osm_metrics, pop_features)
            features.update(interaction_features)
            
            # Cache the result
            self._save_to_cache(cache_key, features)
            
        except Exception as e:
            logger.error(f"Failed to extract comprehensive features: {e}")
            features = {}
        
        return features
    
    def _calculate_interaction_features(self, osm_metrics: Dict[str, float], 
                                      pop_features: Dict[str, float]) -> Dict[str, float]:
        """Calculate interaction features between OSM and population data"""
        interaction_features = {}
        
        try:
            # Population density interactions with amenities
            pop_density = pop_features.get('population_density_km2', 0)
            
            # Key amenity types that interact with population
            key_amenities = ['restaurants', 'shops', 'supermarkets', 'schools', 'parks', 'offices']
            
            for amenity in key_amenities:
                count_key = f'{amenity}_count'
                if count_key in osm_metrics and pop_density > 0:
                    # Amenity per capita
                    amenity_count = osm_metrics[count_key]
                    interaction_features[f'{amenity}_per_1000_people'] = (
                        amenity_count / pop_density * 1000 if pop_density > 0 else 0
                    )
                    
                    # Population-weighted amenity accessibility
                    interaction_features[f'pop_weighted_{amenity}_accessibility'] = (
                        amenity_count * pop_density / 1000
                    )
            
            # Population concentration vs amenity diversity
            pop_entropy = pop_features.get('population_entropy', 0)
            amenity_diversity = sum(1 for key in osm_metrics.keys() 
                                  if key.endswith('_count') and osm_metrics[key] > 0)
            
            interaction_features['amenity_diversity_score'] = amenity_diversity
            interaction_features['pop_amenity_balance'] = (
                pop_entropy * amenity_diversity if amenity_diversity > 0 else 0
            )
            
            # Transit accessibility in population context
            transit_features = ['bus_stops_count', 'train_stations_count']
            total_transit = sum(osm_metrics.get(feature, 0) for feature in transit_features)
            
            if total_transit > 0 and pop_density > 0:
                interaction_features['transit_per_1000_people'] = total_transit / pop_density * 1000
                interaction_features['transit_population_ratio'] = total_transit / (pop_density / 1000)
            else:
                interaction_features['transit_per_1000_people'] = 0
                interaction_features['transit_population_ratio'] = 0
            
        except Exception as e:
            logger.warning(f"Failed to calculate interaction features: {e}")
        
        return interaction_features
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, any]]:
        """Load features from cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, features: Dict[str, any]):
        """Save features to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            # Convert numpy types to Python native types for JSON serialization
            serializable_features = {}
            for key, value in features.items():
                if isinstance(value, (np.integer, np.int64)):
                    serializable_features[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    serializable_features[key] = float(value)
                else:
                    serializable_features[key] = value
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_features, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

if __name__ == "__main__":
    # Example usage
    print("🏙️ Population Feature Extractor Demo")
    
    # Initialize extractor
    extractor = PopulationFeatureExtractor()
    
    # Get summary
    summary = extractor.get_population_summary()
    print(f"\n📊 Population Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test feature extraction
    test_lat, test_lon = 46.5197, 6.6323  # Lausanne area
    
    print(f"\n🎯 Extracting population features around ({test_lat}, {test_lon})...")
    features = extractor.extract_population_features_around_station(test_lat, test_lon, radius_m=500)
    
    print(f"\n📋 Population Features Found:")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test enhanced extractor
    print(f"\n🚀 Testing Enhanced OSM + Population Extractor...")
    enhanced_extractor = EnhancedOSMFeatureExtractor()
    
    comprehensive_features = enhanced_extractor.extract_comprehensive_features(
        test_lat, test_lon, radius_m=500
    )
    
    print(f"\n📈 Comprehensive Features Summary:")
    osm_count = sum(1 for key in comprehensive_features.keys() if key.startswith('osm_'))
    pop_count = sum(1 for key in comprehensive_features.keys() if key.startswith('pop_'))
    interaction_count = len(comprehensive_features) - osm_count - pop_count
    
    print(f"  OSM features: {osm_count}")
    print(f"  Population features: {pop_count}")
    print(f"  Interaction features: {interaction_count}")
    print(f"  Total features: {len(comprehensive_features)}")
