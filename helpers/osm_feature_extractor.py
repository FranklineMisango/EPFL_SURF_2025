"""
OpenStreetMap Feature Extractor for Bike Flow Prediction
Extracts various urban features that influence bike flows: hotels, banks, industrial zones, etc.
"""

import requests
import json
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import math
import overpy
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@dataclass
class OSMFeature:
    """Represents an OSM feature with its properties"""
    feature_type: str
    lat: float
    lon: float
    tags: Dict
    distance_to_station: float = 0.0
    
class OSMFeatureExtractor:
    """Extract various urban features from OpenStreetMap that influence bike flows"""
    
    def __init__(self, cache_dir: str = "cache/osm_features"):
        self.cache_dir = cache_dir
        self.api = overpy.Overpass()
        self.feature_definitions = self._define_feature_types()
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        import os
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _define_feature_types(self) -> Dict[str, Dict]:
        """Define OSM feature types that influence bike flows"""
        return {
            # Commercial & Services
            'hotels': {
                'query': '["tourism"="hotel"]',
                'description': 'Hotels and accommodations',
                'influence': 'Tourist destinations, temporary stays'
            },
            'restaurants': {
                'query': '["amenity"="restaurant"]',
                'description': 'Restaurants and dining',
                'influence': 'Meal destinations, social gatherings'
            },
            'cafes': {
                'query': '["amenity"="cafe"]',
                'description': 'Cafes and coffee shops',
                'influence': 'Work meetings, casual stops'
            },
            'banks': {
                'query': '["amenity"="bank"]',
                'description': 'Banks and financial services',
                'influence': 'Financial errands, business trips'
            },
            'atms': {
                'query': '["amenity"="atm"]',
                'description': 'ATM machines',
                'influence': 'Quick financial stops'
            },
            'shops': {
                'query': '["shop"]',
                'description': 'General retail shops',
                'influence': 'Shopping trips, errands'
            },
            'supermarkets': {
                'query': '["shop"="supermarket"]',
                'description': 'Supermarkets and grocery stores',
                'influence': 'Regular shopping, daily needs'
            },
            'pharmacies': {
                'query': '["amenity"="pharmacy"]',
                'description': 'Pharmacies and drugstores',
                'influence': 'Health-related trips'
            },
            
            # Transportation
            'bus_stops': {
                'query': '["highway"="bus_stop"]',
                'description': 'Bus stops',
                'influence': 'Multimodal transport hubs'
            },
            'train_stations': {
                'query': '["railway"="station"]',
                'description': 'Train stations',
                'influence': 'Major transport hubs'
            },
            'parking': {
                'query': '["amenity"="parking"]',
                'description': 'Parking areas',
                'influence': 'Car-bike transfer points'
            },
            
            # Education & Culture
            'schools': {
                'query': '["amenity"="school"]',
                'description': 'Schools',
                'influence': 'Student commutes, family trips'
            },
            'universities': {
                'query': '["amenity"="university"]',
                'description': 'Universities',
                'influence': 'Student/staff commutes'
            },
            'libraries': {
                'query': '["amenity"="library"]',
                'description': 'Libraries',
                'influence': 'Study destinations, quiet work'
            },
            'museums': {
                'query': '["tourism"="museum"]',
                'description': 'Museums',
                'influence': 'Cultural visits, tourism'
            },
            
            # Healthcare
            'hospitals': {
                'query': '["amenity"="hospital"]',
                'description': 'Hospitals',
                'influence': 'Medical appointments, emergencies'
            },
            'clinics': {
                'query': '["amenity"="clinic"]',
                'description': 'Medical clinics',
                'influence': 'Healthcare visits'
            },
            'dentists': {
                'query': '["amenity"="dentist"]',
                'description': 'Dental offices',
                'influence': 'Medical appointments'
            },
            
            # Recreation & Sports
            'parks': {
                'query': '["leisure"="park"]',
                'description': 'Parks and green spaces',
                'influence': 'Recreation, exercise, relaxation'
            },
            'sports_centres': {
                'query': '["leisure"="sports_centre"]',
                'description': 'Sports centers and gyms',
                'influence': 'Fitness activities, sports'
            },
            'swimming_pools': {
                'query': '["leisure"="swimming_pool"]',
                'description': 'Swimming pools',
                'influence': 'Recreation, fitness'
            },
            'playgrounds': {
                'query': '["leisure"="playground"]',
                'description': 'Playgrounds',
                'influence': 'Family activities'
            },
            
            # Work & Business
            'offices': {
                'query': '["office"]',
                'description': 'Office buildings',
                'influence': 'Work commutes, business meetings'
            },
            'coworking': {
                'query': '["amenity"="coworking_space"]',
                'description': 'Coworking spaces',
                'influence': 'Flexible work locations'
            },
            
            # Zones (Land Use)
            'residential': {
                'query': '["landuse"="residential"]',
                'description': 'Residential areas',
                'influence': 'Home origins/destinations'
            },
            'commercial': {
                'query': '["landuse"="commercial"]',
                'description': 'Commercial zones',
                'influence': 'Business activities'
            },
            'industrial': {
                'query': '["landuse"="industrial"]',
                'description': 'Industrial zones',
                'influence': 'Work commutes, logistics'
            },
            'retail': {
                'query': '["landuse"="retail"]',
                'description': 'Retail areas',
                'influence': 'Shopping destinations'
            },
            
            # Entertainment
            'bars': {
                'query': '["amenity"="bar"]',
                'description': 'Bars and pubs',
                'influence': 'Evening entertainment'
            },
            'nightclubs': {
                'query': '["amenity"="nightclub"]',
                'description': 'Nightclubs',
                'influence': 'Night entertainment'
            },
            'cinemas': {
                'query': '["amenity"="cinema"]',
                'description': 'Movie theaters',
                'influence': 'Entertainment destinations'
            },
            'theatres': {
                'query': '["amenity"="theatre"]',
                'description': 'Theaters',
                'influence': 'Cultural entertainment'
            }
        }
    
    def extract_features_around_station(self, lat: float, lon: float, radius_m: int = 500) -> Dict[str, List[OSMFeature]]:
        """Extract all feature types around a station within given radius"""
        features = {}
        
        logger.info(f"Extracting features around ({lat:.6f}, {lon:.6f}) within {radius_m}m")
        
        for feature_type, definition in self.feature_definitions.items():
            try:
                features[feature_type] = self._extract_single_feature_type(
                    lat, lon, radius_m, feature_type, definition['query']
                )
                logger.info(f"Found {len(features[feature_type])} {feature_type}")
                
                # Rate limiting to avoid overwhelming OSM servers
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to extract {feature_type}: {e}")
                features[feature_type] = []
        
        return features
    
    def _extract_single_feature_type(self, lat: float, lon: float, radius_m: int, 
                                   feature_type: str, query: str) -> List[OSMFeature]:
        """Extract a single feature type using Overpass API"""
        
        # Check cache first
        cache_key = f"{lat:.6f}_{lon:.6f}_{radius_m}_{feature_type}"
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Build Overpass query
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node{query}(around:{radius_m},{lat},{lon});
          way{query}(around:{radius_m},{lat},{lon});
          relation{query}(around:{radius_m},{lat},{lon});
        );
        out center;
        """
        
        try:
            result = self.api.query(overpass_query)
            features = []
            
            # Process nodes
            for node in result.nodes:
                feature = OSMFeature(
                    feature_type=feature_type,
                    lat=float(node.lat),
                    lon=float(node.lon),
                    tags=dict(node.tags),
                    distance_to_station=haversine_distance(lat, lon, node.lat, node.lon)
                )
                features.append(feature)
            
            # Process ways (use center point)
            for way in result.ways:
                if hasattr(way, 'center_lat') and hasattr(way, 'center_lon'):
                    feature = OSMFeature(
                        feature_type=feature_type,
                        lat=float(way.center_lat),
                        lon=float(way.center_lon),
                        tags=dict(way.tags),
                        distance_to_station=haversine_distance(lat, lon, way.center_lat, way.center_lon)
                    )
                    features.append(feature)
            
            # Process relations (use center point if available)
            for relation in result.relations:
                if hasattr(relation, 'center_lat') and hasattr(relation, 'center_lon'):
                    feature = OSMFeature(
                        feature_type=feature_type,
                        lat=float(relation.center_lat),
                        lon=float(relation.center_lon),
                        tags=dict(relation.tags),
                        distance_to_station=haversine_distance(lat, lon, relation.center_lat, relation.center_lon)
                    )
                    features.append(feature)
            
            # Cache the result
            self._save_to_cache(cache_key, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Overpass query failed for {feature_type}: {e}")
            return []
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[OSMFeature]]:
        """Load features from cache"""
        import os
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                features = []
                for item in data:
                    feature = OSMFeature(
                        feature_type=item['feature_type'],
                        lat=item['lat'],
                        lon=item['lon'],
                        tags=item['tags'],
                        distance_to_station=item['distance_to_station']
                    )
                    features.append(feature)
                
                return features
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, features: List[OSMFeature]):
        """Save features to cache"""
        import os
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            data = []
            for feature in features:
                data.append({
                    'feature_type': feature.feature_type,
                    'lat': feature.lat,
                    'lon': feature.lon,
                    'tags': feature.tags,
                    'distance_to_station': feature.distance_to_station
                })
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def compute_feature_metrics(self, features: Dict[str, List[OSMFeature]], 
                              distance_bands: List[int] = [100, 250, 500]) -> Dict[str, float]:
        """Compute aggregated metrics for features around a station"""
        metrics = {}
        
        for feature_type, feature_list in features.items():
            # Basic counts
            metrics[f'{feature_type}_count'] = len(feature_list)
            
            if feature_list:
                # Distance statistics
                distances = [f.distance_to_station for f in feature_list]
                metrics[f'{feature_type}_min_distance'] = min(distances)
                metrics[f'{feature_type}_avg_distance'] = np.mean(distances)
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
                metrics[f'{feature_type}_min_distance'] = 999999
                metrics[f'{feature_type}_avg_distance'] = 999999
                metrics[f'{feature_type}_closest_count'] = 0
                metrics[f'{feature_type}_density'] = 0
                
                for band in distance_bands:
                    metrics[f'{feature_type}_within_{band}m'] = 0
        
        return metrics
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all feature types"""
        return {
            feature_type: definition['description'] 
            for feature_type, definition in self.feature_definitions.items()
        }
    
    def get_feature_influences(self) -> Dict[str, str]:
        """Get influence descriptions for all feature types"""
        return {
            feature_type: definition['influence'] 
            for feature_type, definition in self.feature_definitions.items()
        }

class FeatureInfluenceAnalyzer:
    """Analyze the influence of different features on bike flow destinations"""
    
    def __init__(self):
        self.feature_importance = {}
        self.correlation_matrix = None
        
    def analyze_feature_influence(self, station_features: pd.DataFrame, 
                                flow_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze which features most influence destination choices"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        import scipy.stats as stats
        
        # Prepare data for analysis
        feature_columns = [col for col in station_features.columns 
                          if col not in ['station_id', 'lat', 'lon']]
        
        # Merge station features with flow data
        flow_with_features = flow_data.merge(
            station_features, 
            left_on='destination_station', 
            right_on='station_id', 
            how='inner'
        )
        
        if len(flow_with_features) == 0:
            return {}
        
        X = flow_with_features[feature_columns].fillna(0)
        y = flow_with_features['flow_volume']
        
        # Feature importance from Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = dict(zip(feature_columns, rf.feature_importances_))
        
        # Correlation analysis
        correlations = {}
        for feature in feature_columns:
            if X[feature].std() > 0:  # Avoid constant features
                corr, p_value = stats.pearsonr(X[feature], y)
                correlations[feature] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Combine importance and correlation
        influence_scores = {}
        for feature in feature_columns:
            rf_importance = feature_importance.get(feature, 0)
            corr_strength = abs(correlations.get(feature, {}).get('correlation', 0))
            
            # Combined score (weighted average)
            influence_scores[feature] = 0.7 * rf_importance + 0.3 * corr_strength
        
        # Sort by influence
        self.feature_importance = dict(
            sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return self.feature_importance
    
    def get_top_influential_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most influential features"""
        return list(self.feature_importance.items())[:n]
    
    def create_influence_report(self, extractor: OSMFeatureExtractor) -> str:
        """Create a detailed influence analysis report"""
        if not self.feature_importance:
            return "No feature influence analysis available. Run analyze_feature_influence first."
        
        report = "# Feature Influence Analysis Report\n\n"
        
        # Top influential features
        report += "## Top 10 Most Influential Features\n\n"
        top_features = self.get_top_influential_features(10)
        
        feature_descriptions = extractor.get_feature_descriptions()
        feature_influences = extractor.get_feature_influences()
        
        for i, (feature, score) in enumerate(top_features, 1):
            # Extract feature type from metric name
            feature_type = feature.split('_')[0]
            description = feature_descriptions.get(feature_type, "Unknown feature")
            influence = feature_influences.get(feature_type, "Unknown influence")
            
            report += f"{i}. **{feature}** (Score: {score:.4f})\n"
            report += f"   - Description: {description}\n"
            report += f"   - Influence: {influence}\n\n"
        
        # Feature categories analysis
        report += "## Feature Category Analysis\n\n"
        category_scores = {}
        
        for feature, score in self.feature_importance.items():
            feature_type = feature.split('_')[0]
            if feature_type not in category_scores:
                category_scores[feature_type] = []
            category_scores[feature_type].append(score)
        
        # Average scores by category
        category_averages = {
            category: np.mean(scores) 
            for category, scores in category_scores.items()
        }
        
        sorted_categories = sorted(category_averages.items(), key=lambda x: x[1], reverse=True)
        
        for category, avg_score in sorted_categories[:15]:  # Top 15 categories
            description = feature_descriptions.get(category, "Unknown")
            report += f"- **{category.title()}**: {avg_score:.4f} - {description}\n"
        
        return report

if __name__ == "__main__":
    # Example usage
    extractor = OSMFeatureExtractor()
    
    # Example coordinates (you would use actual station coordinates)
    test_lat, test_lon = 46.5197, 6.6323  # Lausanne area
    
    print("Extracting OSM features...")
    features = extractor.extract_features_around_station(test_lat, test_lon, radius_m=500)
    
    print("\nFeature Summary:")
    for feature_type, feature_list in features.items():
        print(f"{feature_type}: {len(feature_list)} found")
    
    print("\nComputing metrics...")
    metrics = extractor.compute_feature_metrics(features)
    
    print("\nTop metrics:")
    for key, value in list(metrics.items())[:10]:
        print(f"{key}: {value}")