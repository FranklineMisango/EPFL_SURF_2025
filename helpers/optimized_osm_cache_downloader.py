#!/usr/bin/env python3
"""
Optimized Master OSM Cache Downloader for ML Algorithm Preparation
Comprehensive solution to download, cache, and consolidate OSM features before running ML algorithms.
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import sys
import re
from collections import deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import psutil  # For memory tracking
from tqdm import tqdm  # For progress bars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for OSM cache downloading"""
    cache_base_dir: str = "cache"
    trips_file: str = "data/trips_8days_flat.csv"
    radius_meters: int = 1000
    
    # Download options
    use_traditional_osm: bool = True
    use_osmnx: bool = True
    use_population: bool = True
    
    # Performance settings
    batch_size: int = 20
    rate_limit_seconds: float = 1.0  # Increased default to 1.0 second
    max_workers: int = 4  # Reduced default to 4 for gentler API usage
    force_refresh: bool = False
    
    # Cache management
    consolidate_cache: bool = True
    cleanup_individual_files: bool = False
    create_summary_stats: bool = True
    
    # Memory management
    memory_cache_limit_mb: int = 512  # 512MB memory cache limit
    
    # Advanced options
    use_adaptive_rate_limiting: bool = True
    use_predictive_prefetching: bool = True
    
    # Rate limiting options
    gentle_mode: bool = False  # Extra gentle mode for heavily rate-limited APIs

class DiskCache:
    """Disk-based cache with efficient I/O"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get path for a cache key with directory hashing for scalability"""
        # Use first 2 chars of hash for directory sharding
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        shard_dir = os.path.join(self.cache_dir, key_hash[:2])
        os.makedirs(shard_dir, exist_ok=True)
        return os.path.join(shard_dir, f"{key_hash}.json")
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache"""
        return os.path.exists(self._get_cache_path(key))
    
    def get(self, key: str) -> Optional[Dict]:
        """Get value for key"""
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None
    
    def set(self, key: str, value: Dict) -> bool:
        """Set value for key"""
        path = self._get_cache_path(key)
        try:
            with open(path, 'w') as f:
                json.dump(value, f)
            return True
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
            return False
    
    def get_batch(self, keys: List[str]) -> Dict[str, Dict]:
        """Get multiple values efficiently"""
        result = {}
        for key in keys:
            value = self.get(key)
            if value:
                result[key] = value
        return result

class SmartCacheManager:
    """Hierarchical cache with memory and disk layers"""
    
    def __init__(self, cache_dir: str, memory_limit_mb: int = 512):
        self.memory_cache = {}
        self.disk_cache = DiskCache(cache_dir)
        self.prefetch_queue = deque()
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.memory_usage = 0
        self.hits = {"memory": 0, "disk": 0, "miss": 0}
        # Add batch cache for efficiency
        self._batch_cache = {}
        self._batch_size = 50
    
    def get_batch_from_cache(self, keys: List[str]) -> Tuple[Dict[str, Dict], List[str]]:
        """Get multiple items from cache, return (found_items, missing_keys)"""
        found = {}
        missing = []
        
        for key in keys:
            # Check memory first
            if key in self.memory_cache:
                found[key] = self.memory_cache[key]
                self.hits["memory"] += 1
            else:
                missing.append(key)
        
        # Batch check disk for missing items
        if missing:
            disk_found = self.disk_cache.get_batch(missing)
            for key, value in disk_found.items():
                found[key] = value
                self.hits["disk"] += 1
                self._add_to_memory_cache(key, value)
                missing.remove(key)
        
        # Track misses
        self.hits["miss"] += len(missing)
        return found, missing

    def get(self, key: str) -> Optional[Dict]:
        """Get value from cache hierarchy"""
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            self.hits["memory"] += 1
            return self.memory_cache[key]
        
        # Try disk cache second
        value = self.disk_cache.get(key)
        if value:
            self.hits["disk"] += 1
            # Promote to memory cache if space allows
            self._add_to_memory_cache(key, value)
            return value
        
        self.hits["miss"] += 1
        return None
    
    def set(self, key: str, value: Dict) -> None:
        """Set value in cache hierarchy"""
        # Always save to disk
        self.disk_cache.set(key, value)
        
        # Add to memory if space allows
        self._add_to_memory_cache(key, value)
    
    def _add_to_memory_cache(self, key: str, value: Dict) -> None:
        """Add to memory cache with size management"""
        # Estimate size of the value
        value_size = len(json.dumps(value).encode())
        
        # If would exceed limit, make space
        while self.memory_usage + value_size > self.memory_limit_bytes and self.memory_cache:
            # Remove oldest item (simple LRU strategy)
            oldest_key = next(iter(self.memory_cache))
            oldest_value = self.memory_cache.pop(oldest_key)
            oldest_size = len(json.dumps(oldest_value).encode())
            self.memory_usage -= oldest_size
        
        # Add to memory cache if it fits
        if value_size <= self.memory_limit_bytes:
            self.memory_cache[key] = value
            self.memory_usage += value_size
    
    def prefetch(self, keys: List[str]) -> None:
        """Queue keys for background prefetching"""
        for key in keys:
            if key not in self.memory_cache and key not in self.prefetch_queue:
                self.prefetch_queue.append(key)
    
    def process_prefetch_queue(self, max_items: int = 10) -> int:
        """Process some items from prefetch queue, returns number processed"""
        processed = 0
        while self.prefetch_queue and processed < max_items:
            key = self.prefetch_queue.popleft()
            if key not in self.memory_cache:
                value = self.disk_cache.get(key)
                if value:
                    self._add_to_memory_cache(key, value)
                    processed += 1
        return processed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = sum(self.hits.values())
        return {
            "memory_hits": self.hits["memory"],
            "disk_hits": self.hits["disk"],
            "misses": self.hits["miss"],
            "hit_rate": (self.hits["memory"] + self.hits["disk"]) / max(1, total_requests),
            "memory_hit_rate": self.hits["memory"] / max(1, total_requests),
            "memory_usage_mb": self.memory_usage / (1024 * 1024),
            "memory_limit_mb": self.memory_limit_bytes / (1024 * 1024),
            "items_in_memory": len(self.memory_cache)
        }

class AdaptiveRateLimiter:
    """Adaptive rate limiter for API requests with enhanced error handling for Overpass API"""
    
    def __init__(self, initial_delay: float = 0.5, min_delay: float = 0.1, max_delay: float = 5.0):
        self.delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request = time.time()
        self.success_streak = 0
        self.failure_streak = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.too_many_requests_count = 0
        self.retry_after_seconds = None
        
        # Exponential backoff settings
        self.backoff_factor = 2.0
        self.jitter = 0.05  # Reduced jitter for faster processing
        
        # Skip rate limiting for cache hits
        self.skip_next = False
    
    def skip_rate_limit(self):
        """Skip rate limiting for the next request (used for cache hits)"""
        self.skip_next = True
    
    def __enter__(self):
        """Apply rate limiting before request with retry handling"""
        if self.skip_next:
            self.skip_next = False
            return self
            
        # If we've been told to wait, honor that first
        if self.retry_after_seconds is not None:
            now = time.time()
            wait_until = self.last_request + self.retry_after_seconds
            if now < wait_until:
                wait_time = wait_until - now
                logger.info(f"Respecting API rate limit: waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            self.retry_after_seconds = None
        
        # Standard rate limiting
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        
        # Add small random jitter to avoid synchronized requests
        jitter_seconds = self.delay * self.jitter * (2 * np.random.random() - 1)
        if jitter_seconds > 0:
            time.sleep(jitter_seconds)
            
        self.last_request = time.time()
        self.total_requests += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Adjust rate limit based on success/failure with special handling for rate limit errors"""
        if exc_type is None:
            # Success
            self.success_streak += 1
            self.failure_streak = 0
            
            # Gradually decrease delay after consistent success, but cautiously
            if self.success_streak >= 10:  # Require more successes before decreasing
                self.delay = max(self.min_delay, self.delay * 0.95)  # Decrease more slowly
        else:
            # Failure
            self.failed_requests += 1
            self.failure_streak += 1
            self.success_streak = 0
            
            # Check if it's a rate limit error (for Overpass API)
            rate_limit_error = False
            
            # Common rate limit error messages across different APIs
            rate_limit_patterns = [
                "too many requests",
                "rate limit exceeded",
                "quota exceeded",
                "rate limited",
                "try again later",
                "too many connections",
                "429", # HTTP status code for too many requests
                "slow down"
            ]
            
            if exc_val and isinstance(exc_val, Exception):
                error_msg = str(exc_val).lower()
                for pattern in rate_limit_patterns:
                    if pattern in error_msg:
                        rate_limit_error = True
                        self.too_many_requests_count += 1
                        break
            
            if rate_limit_error:
                # Apply exponential backoff for rate limit errors
                if self.too_many_requests_count == 1:
                    # First rate limit error, double the current delay
                    self.delay = min(self.max_delay, self.delay * 2.0)
                    self.retry_after_seconds = self.delay
                    logger.warning(f"Rate limit hit! Increasing delay to {self.delay:.2f}s")
                else:
                    # Subsequent rate limit errors, apply exponential backoff
                    backoff = min(self.max_delay, 
                                  self.delay * (self.backoff_factor ** min(self.too_many_requests_count, 5)))
                    self.delay = backoff
                    self.retry_after_seconds = backoff
                    logger.warning(f"Multiple rate limits! Backing off for {backoff:.2f}s")
            else:
                # Other non-rate-limit errors - increase delay but less aggressively
                self.delay = min(self.max_delay, self.delay * 1.2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "current_delay": self.delay,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "too_many_requests_count": self.too_many_requests_count,
            "success_rate": 1 - (self.failed_requests / max(1, self.total_requests)),
            "effective_throughput": self.total_requests / (time.time() - self.last_request + 0.001)
        }

class FeatureOptimizer:
    """Optimizes feature dataframes for ML processing"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage and processing efficiency"""
        # Make a copy to avoid modifying the original
        df_opt = df.copy()
        
        # Type optimization
        for col in df_opt.columns:
            if df_opt[col].dtype == 'float64':
                # Try to downcast floats
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
            elif df_opt[col].dtype == 'int64':
                # Try to downcast integers
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
            elif df_opt[col].dtype == 'object' and col != 'station_id':
                # Try to convert string columns to categorical for memory efficiency
                if df_opt[col].nunique() < len(df_opt) // 2:
                    df_opt[col] = df_opt[col].astype('category')
        
        # Feature-specific optimizations
        count_cols = [c for c in df_opt.columns if c.startswith('count_') or c.endswith('_count')]
        for col in count_cols:
            # Most count columns can be unsigned int
            if df_opt[col].min() >= 0:
                df_opt[col] = pd.to_numeric(df_opt[col], downcast='unsigned')
        
        density_cols = [c for c in df_opt.columns if 'density' in c or 'ratio' in c]
        for col in density_cols:
            # Most density columns have many zeros
            zeros_ratio = (df_opt[col] == 0).mean()
            if zeros_ratio > 0.7:  # If more than 70% zeros
                df_opt[col] = df_opt[col].astype(pd.SparseDtype("float", 0.0))
        
        return df_opt
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        # Different strategies for different column types
        count_cols = [c for c in df.columns if c.startswith('count_') or c.endswith('_count')]
        ratio_cols = [c for c in df.columns if 'ratio' in c or 'density' in c]
        
        # Fill counts with 0
        df[count_cols] = df[count_cols].fillna(0)
        
        # Fill ratios with median (more robust than mean)
        for col in ratio_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        
        # For remaining columns, use standard imputation
        remaining_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                         if c not in count_cols and c not in ratio_cols]
        df[remaining_cols] = df[remaining_cols].fillna(df[remaining_cols].median())
        
        return df

class MasterOSMCacheDownloader:
    """Master downloader that orchestrates all OSM feature downloading and caching"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.station_coords = {}
        self.download_stats = {
            'total_stations': 0,
            'successful_stations': 0,
            'failed_stations': 0,
            'start_time': None,
            'end_time': None,
            'errors': [],
            'performance_metrics': {},
        }
        
        # Set up cache directories and managers
        self._setup_cache_directories()
        
        # Initialize cache managers
        self.cache_manager = SmartCacheManager(
            os.path.join(config.cache_base_dir, 'smart_cache'),
            memory_limit_mb=config.memory_cache_limit_mb
        )
        
        # Initialize rate limiters with more conservative defaults for OSM
        if config.gentle_mode:
            # Extra gentle mode for heavily rate-limited APIs
            self.rate_limiters = {
                'osm': AdaptiveRateLimiter(
                    initial_delay=max(2.0, config.rate_limit_seconds * 4),
                    min_delay=1.0,  # Never go below 1 second in gentle mode
                    max_delay=10.0  # Allow longer backoff periods
                ),
                'population': AdaptiveRateLimiter(
                    initial_delay=max(1.5, config.rate_limit_seconds * 3),
                    min_delay=0.5,
                    max_delay=8.0
                ),
                'comprehensive': AdaptiveRateLimiter(
                    initial_delay=max(3.0, config.rate_limit_seconds * 6),
                    min_delay=2.0,
                    max_delay=15.0
                )
            }
            logger.info("🐢 Using extra gentle mode for rate limiting")
        else:
            # Normal mode
            self.rate_limiters = {
                'osm': AdaptiveRateLimiter(
                    initial_delay=max(1.0, config.rate_limit_seconds * 2)
                ),
                'population': AdaptiveRateLimiter(
                    initial_delay=config.rate_limit_seconds
                ),
                'comprehensive': AdaptiveRateLimiter(
                    initial_delay=max(1.5, config.rate_limit_seconds * 3)
                )
            }
        
        # Initialize feature extractors
        self._initialize_extractors()
        
        # Initialize progress bar
        self.pbar = None
    
    def _setup_cache_directories(self):
        """Set up all required cache directories"""
        cache_dirs = [
            self.config.cache_base_dir,
            os.path.join(self.config.cache_base_dir, 'osm_features'),
            os.path.join(self.config.cache_base_dir, 'population_features'),
            os.path.join(self.config.cache_base_dir, 'comprehensive_osm'),
            os.path.join(self.config.cache_base_dir, 'consolidated'),
            os.path.join(self.config.cache_base_dir, 'ml_ready'),
            os.path.join(self.config.cache_base_dir, 'smart_cache')
        ]
        
        for cache_dir in cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"✅ Cache directory ready: {cache_dir}")
    
    def _initialize_extractors(self):
        """Initialize feature extractors with error handling"""
        self.extractors = {}
        
        # Traditional OSM extractor
        if self.config.use_traditional_osm:
            try:
                from osm_feature_extractor import OSMFeatureExtractor
                self.extractors['osm'] = OSMFeatureExtractor(
                    cache_dir=os.path.join(self.config.cache_base_dir, 'osm_features')
                )
                logger.info("✅ Traditional OSM extractor initialized")
            except ImportError as e:
                logger.error(f"❌ Failed to initialize OSM extractor: {e}")
                self.config.use_traditional_osm = False
        
        # Population extractor
        if self.config.use_population:
            try:
                from population_feature_extractor import PopulationFeatureExtractor, EnhancedOSMFeatureExtractor
                self.extractors['population'] = PopulationFeatureExtractor()
                self.extractors['enhanced'] = EnhancedOSMFeatureExtractor()
                logger.info("✅ Population extractors initialized")
            except ImportError as e:
                logger.error(f"❌ Failed to initialize population extractors: {e}")
                self.config.use_population = False
        
        # OSMnx extractor (comprehensive downloader)
        if self.config.use_osmnx:
            try:
                from comprehensive_osm_downloader import ComprehensiveOSMDownloader, OSMDownloadConfig
                osmnx_config = OSMDownloadConfig(
                    cache_dir=os.path.join(self.config.cache_base_dir, 'comprehensive_osm'),
                    radius_meters=self.config.radius_meters,
                    use_osmnx=True,
                    use_overpass=False,  # Avoid duplicate with traditional
                    rate_limit_seconds=self.config.rate_limit_seconds,
                    force_refresh=self.config.force_refresh
                )
                self.extractors['comprehensive'] = ComprehensiveOSMDownloader(osmnx_config)
                logger.info("✅ Comprehensive OSM extractor initialized")
            except ImportError as e:
                logger.error(f"❌ Failed to initialize comprehensive extractor: {e}")
                self.config.use_osmnx = False
    
    def load_station_coordinates(self) -> bool:
        """Load station coordinates from trips data using vectorized operations"""
        logger.info(f"📍 Loading station coordinates from {self.config.trips_file}")
        
        try:
            if not os.path.exists(self.config.trips_file):
                logger.error(f"❌ Trips file not found: {self.config.trips_file}")
                return False
            
            # Read CSV using pandas
            trips_df = pd.read_csv(self.config.trips_file)
            logger.info(f"📊 Loaded {len(trips_df)} trips")
            
            # Vectorized coordinate parsing using regex
            coord_regex = r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)'
            
            # Extract start and end coordinates using vectorized string operations
            start_coords = trips_df['start_coords'].str.extract(coord_regex)
            end_coords = trips_df['end_coords'].str.extract(coord_regex)
            
            # Convert to appropriate types
            trips_df['start_lat'] = pd.to_numeric(start_coords[0], errors='coerce')
            trips_df['start_lon'] = pd.to_numeric(start_coords[1], errors='coerce')
            trips_df['end_lat'] = pd.to_numeric(end_coords[0], errors='coerce')
            trips_df['end_lon'] = pd.to_numeric(end_coords[1], errors='coerce')
            
            # Create separate dataframes for start and end stations
            start_stations = trips_df[['start_station_id', 'start_lat', 'start_lon']].rename(
                columns={'start_station_id': 'station_id', 'start_lat': 'lat', 'start_lon': 'lon'}
            ).dropna()
            
            end_stations = trips_df[['end_station_id', 'end_lat', 'end_lon']].rename(
                columns={'end_station_id': 'station_id', 'end_lat': 'lat', 'end_lon': 'lon'}
            ).dropna()
            
            # Concatenate and drop duplicates (keep first occurrence)
            all_stations = pd.concat([start_stations, end_stations]).drop_duplicates(subset='station_id')
            
            # Convert to dictionary for fast lookup
            self.station_coords = {
                row['station_id']: (row['lat'], row['lon']) 
                for _, row in all_stations.iterrows()
            }
            
            self.download_stats['total_stations'] = len(self.station_coords)
            logger.info(f"✅ Loaded coordinates for {len(self.station_coords)} unique stations")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load station coordinates: {e}")
            return False
    
    def _process_station(self, extractor_type: str, station_item) -> Tuple[str, str, Optional[Dict]]:
        """Process a single station for a specific extractor type"""
        station_id, (lat, lon) = station_item
        cache_key = f"{extractor_type}_{station_id}"
        
        # Check cache first
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            return extractor_type, station_id, cached_data
        
        # Apply rate limiting based on extractor type
        rate_limiter = self.rate_limiters.get(extractor_type, self.rate_limiters['osm'])
        
        try:
            with rate_limiter:
                # Extract features based on extractor type
                if extractor_type == 'osm':
                    features = self.extractors['osm'].extract_features_around_station(
                        lat, lon, radius_m=self.config.radius_meters
                    )
                    metrics = self.extractors['osm'].compute_feature_metrics(features)
                    result = metrics
                
                elif extractor_type == 'population':
                    pop_features = self.extractors['population'].extract_population_features_around_station(
                        lat, lon, radius_m=self.config.radius_meters
                    )
                    enhanced_features = self.extractors['enhanced'].extract_comprehensive_features(
                        lat, lon, radius_m=self.config.radius_meters
                    )
                    result = {**pop_features, **enhanced_features}
                
                elif extractor_type == 'comprehensive':
                    # This needs to match your comprehensive extractor interface
                    result = self.extractors['comprehensive'].download_features_for_station(
                        station_id, lat, lon, radius_m=self.config.radius_meters
                    )
                
                else:
                    logger.error(f"Unknown extractor type: {extractor_type}")
                    return extractor_type, station_id, None
            
            # Cache the result
            self.cache_manager.set(cache_key, result)
            
            return extractor_type, station_id, result
            
        except Exception as e:
            error_msg = f"Station {station_id} ({extractor_type}): {str(e)}"
            self.download_stats['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return extractor_type, station_id, None
    
    def _process_station_batch(self, extractor_type: str, station_batch: List[Tuple]) -> List[Tuple[str, str, Optional[Dict]]]:
        """Process a batch of stations more efficiently"""
        results = []
        
        # Check cache for entire batch first
        cache_keys = [f"{extractor_type}_{station_id}" for station_id, _ in station_batch]
        cached_items, missing_keys = self.cache_manager.get_batch_from_cache(cache_keys)
        
        # Process cached items (no API calls needed)
        for i, (station_id, coords) in enumerate(station_batch):
            cache_key = f"{extractor_type}_{station_id}"
            if cache_key in cached_items:
                results.append((extractor_type, station_id, cached_items[cache_key]))
                # Skip rate limiting for cache hits
                continue
        
        # Process items that need API calls
        missing_stations = [
            (station_id, coords) for station_id, coords in station_batch 
            if f"{extractor_type}_{station_id}" in missing_keys
        ]
        
        rate_limiter = self.rate_limiters.get(extractor_type, self.rate_limiters['osm'])
        
        for station_id, (lat, lon) in missing_stations:
            try:
                with rate_limiter:
                    # Extract features based on extractor type
                    if extractor_type == 'osm':
                        features = self.extractors['osm'].extract_features_around_station(
                            lat, lon, radius_m=self.config.radius_meters
                        )
                        metrics = self.extractors['osm'].compute_feature_metrics(features)
                        result = metrics
                    
                    elif extractor_type == 'population':
                        pop_features = self.extractors['population'].extract_population_features_around_station(
                            lat, lon, radius_m=self.config.radius_meters
                        )
                        enhanced_features = self.extractors['enhanced'].extract_comprehensive_features(
                            lat, lon, radius_m=self.config.radius_meters
                        )
                        result = {**pop_features, **enhanced_features}
                    
                    elif extractor_type == 'comprehensive':
                        result = self.extractors['comprehensive'].download_features_for_station(
                            station_id, lat, lon, radius_m=self.config.radius_meters
                        )
                    
                    else:
                        logger.error(f"Unknown extractor type: {extractor_type}")
                        result = None
                
                # Cache the result
                if result is not None:
                    cache_key = f"{extractor_type}_{station_id}"
                    self.cache_manager.set(cache_key, result)
                
                results.append((extractor_type, station_id, result))
                
            except Exception as e:
                error_msg = f"Station {station_id} ({extractor_type}): {str(e)}"
                self.download_stats['errors'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                results.append((extractor_type, station_id, None))
        
        return results
    
    def _batch_download(self, extractor_type: str) -> Dict[str, Dict]:
        """Download features in optimized parallel batches"""
        if extractor_type not in self.extractors:
            logger.warning(f"⚠️ Extractor {extractor_type} not initialized, skipping")
            return {}
        
        logger.info(f"🔄 Starting optimized download for {extractor_type} extractor")
        
        all_features = {}
        station_items = list(self.station_coords.items())
        
        # Create batches for more efficient processing
        batch_size = min(self.config.batch_size, 10)  # Smaller batches for better progress tracking
        station_batches = [
            station_items[i:i + batch_size] 
            for i in range(0, len(station_items), batch_size)
        ]
        
        # Initialize progress bar for this extractor
        pbar = tqdm(
            total=len(station_items),
            desc=f"Processing {extractor_type}",
            unit="stations",
            leave=True
        )
        
        # Process batches with reduced parallelism for better API behavior
        max_workers = min(self.config.max_workers, 2)  # Reduced for better stability
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_station_batch, extractor_type, batch): batch
                for batch in station_batches
            }
            
            # Process completed batches
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=300)  # 5 minute timeout per batch
                    
                    for ext_type, station_id, result in batch_results:
                        if result is not None:
                            all_features[station_id] = result
                            self.download_stats['successful_stations'] += 1
                        else:
                            self.download_stats['failed_stations'] += 1
                        
                        # Update progress
                        pbar.update(1)
                        
                        # Show cache efficiency
                        cache_stats = self.cache_manager.get_stats()
                        pbar.set_postfix({
                            'success': len(all_features),
                            'cache_hit': f"{cache_stats['hit_rate']*100:.1f}%"
                        })
                
                except Exception as e:
                    logger.error(f"❌ Batch processing failed: {e}")
                    # Update progress for failed batch
                    pbar.update(len(batch))
        
        pbar.close()
        
        logger.info(f"✅ {extractor_type} download completed: {len(all_features)} successful stations")
        
        # Show final cache statistics
        cache_stats = self.cache_manager.get_stats()
        logger.info(f"📊 Cache performance: {cache_stats['hit_rate']*100:.1f}% hit rate, " +
                   f"{cache_stats['memory_usage_mb']:.1f}MB memory used")
        
        return all_features
    
    def download_all_features(self) -> bool:
        """Download all OSM features with optimized parallel processing"""
        if not self.station_coords:
            logger.error("❌ No station coordinates loaded. Run load_station_coordinates() first.")
            return False
        
        self.download_stats['start_time'] = time.time()
        logger.info("🚀 Starting optimized OSM feature download...")
        logger.info(f"📍 Processing {len(self.station_coords)} stations with radius {self.config.radius_meters}m")
        
        # Track memory usage before download
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Dictionary to store all features
        all_features = {
            'osm': {},
            'population': {},
            'comprehensive': {}
        }
        
        # Download features sequentially by extractor type to avoid API conflicts
        extractors_to_run = []
        if self.config.use_traditional_osm and 'osm' in self.extractors:
            extractors_to_run.append('osm')
        if self.config.use_population and 'population' in self.extractors:
            extractors_to_run.append('population')
        if self.config.use_osmnx and 'comprehensive' in self.extractors:
            extractors_to_run.append('comprehensive')
        
        for extractor_type in extractors_to_run:
            logger.info(f"🔄 Starting {extractor_type} feature extraction...")
            start_time = time.time()
            
            features = self._batch_download(extractor_type)
            all_features[extractor_type] = features
            
            # Save intermediate results
            cache_file = os.path.join(
                self.config.cache_base_dir, 
                'consolidated', 
                f'{extractor_type}_features.json'
            )
            if extractor_type == 'osm':
                cache_file = os.path.join(
                    self.config.cache_base_dir, 
                    'consolidated', 
                    'traditional_osm_features.json'
                )
            elif extractor_type == 'comprehensive':
                cache_file = os.path.join(
                    self.config.cache_base_dir, 
                    'consolidated', 
                    'comprehensive_osm_features.json'
                )
            
            self._save_cache(cache_file, features)
            
            duration = time.time() - start_time
            rate = len(features) / max(1, duration)
            logger.info(f"✅ {extractor_type} completed: {len(features)} stations in {duration:.1f}s ({rate:.2f} stations/sec)")
        
        # Close progress bar if it exists
        if self.pbar is not None:
            self.pbar.close()
        
        self.download_stats['end_time'] = time.time()
        
        # Track memory after download
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.download_stats['performance_metrics']['memory_increase_mb'] = final_memory - initial_memory
        
        # Calculate performance metrics
        total_duration = self.download_stats['end_time'] - self.download_stats['start_time']
        self.download_stats['performance_metrics'].update({
            'stations_per_second': len(self.station_coords) / total_duration,
            'cache_performance': self.cache_manager.get_stats(),
            'rate_limiters': {k: v.get_stats() for k, v in self.rate_limiters.items()},
            'download_duration_seconds': total_duration
        })
        
        # Consolidate all caches
        if self.config.consolidate_cache:
            success = self._consolidate_all_caches()
            if not success:
                return False
        
        # Create ML-ready features
        success = self._create_ml_ready_features()
        if not success:
            return False
        
        # Generate summary
        if self.config.create_summary_stats:
            self._generate_download_summary()
        
        return True
    
    def _consolidate_all_caches(self) -> bool:
        """Consolidate all individual cache files into unified caches"""
        logger.info("🔄 Consolidating all caches...")
        
        try:
            # Create ML-ready consolidated cache
            ml_cache_file = os.path.join(self.config.cache_base_dir, 'ml_ready', 'all_features.json')
            
            all_station_features = {}
            
            # Load traditional OSM features
            traditional_cache = os.path.join(self.config.cache_base_dir, 'consolidated', 'traditional_osm_features.json')
            if os.path.exists(traditional_cache):
                with open(traditional_cache, 'r') as f:
                    traditional_data = json.load(f)
                    for station_id, features in traditional_data.items():
                        if station_id not in all_station_features:
                            all_station_features[station_id] = {}
                        # Add prefix to avoid conflicts
                        for key, value in features.items():
                            all_station_features[station_id][f'trad_{key}'] = value
                logger.info(f"📊 Loaded traditional features for {len(traditional_data)} stations")
            
            # Load population features
            population_cache = os.path.join(self.config.cache_base_dir, 'consolidated', 'population_features.json')
            if os.path.exists(population_cache):
                with open(population_cache, 'r') as f:
                    population_data = json.load(f)
                    for station_id, features in population_data.items():
                        if station_id not in all_station_features:
                            all_station_features[station_id] = {}
                        # Add prefix to avoid conflicts
                        for key, value in features.items():
                            all_station_features[station_id][f'pop_{key}'] = value
                logger.info(f"📊 Loaded population features for {len(population_data)} stations")
            
            # Load comprehensive OSM features
            comprehensive_cache = os.path.join(self.config.cache_base_dir, 'consolidated', 'comprehensive_osm_features.json')
            if os.path.exists(comprehensive_cache):
                with open(comprehensive_cache, 'r') as f:
                    comprehensive_data = json.load(f)
                    for station_id, features in comprehensive_data.items():
                        if station_id not in all_station_features:
                            all_station_features[station_id] = {}
                        # Add prefix to avoid conflicts
                        for key, value in features.items():
                            all_station_features[station_id][f'comp_{key}'] = value
                logger.info(f"📊 Loaded comprehensive features for {len(comprehensive_data)} stations")
            
            # Add station coordinates
            for station_id, (lat, lon) in self.station_coords.items():
                station_id_str = str(station_id)
                if station_id_str not in all_station_features:
                    all_station_features[station_id_str] = {}
                all_station_features[station_id_str]['station_id'] = station_id
                all_station_features[station_id_str]['lat'] = lat
                all_station_features[station_id_str]['lon'] = lon
            
            # Save consolidated cache
            self._save_cache(ml_cache_file, all_station_features)
            
            logger.info(f"✅ Consolidated cache created with {len(all_station_features)} stations")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache consolidation failed: {e}")
            return False
    
    def _create_ml_ready_features(self) -> bool:
        """Create ML-ready feature matrix with optimization"""
        logger.info("🤖 Creating ML-ready feature matrix...")
        
        try:
            # Load consolidated features
            ml_cache_file = os.path.join(self.config.cache_base_dir, 'ml_ready', 'all_features.json')
            
            if not os.path.exists(ml_cache_file):
                logger.error("❌ Consolidated cache not found. Run consolidation first.")
                return False
            
            with open(ml_cache_file, 'r') as f:
                all_features = json.load(f)
            
            # Convert to DataFrame
            features_list = []
            for station_id, features in all_features.items():
                feature_row = {'station_id': station_id}
                feature_row.update(features)
                features_list.append(feature_row)
            
            features_df = pd.DataFrame(features_list)
            
            # Apply feature optimization
            logger.info("🔧 Optimizing feature dataframe...")
            features_df = FeatureOptimizer.handle_missing_values(features_df)
            optimized_df = FeatureOptimizer.optimize_dataframe(features_df)
            
            # Calculate memory savings
            original_size = features_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            optimized_size = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            
            logger.info(f"📊 Memory optimization: {original_size:.2f}MB -> {optimized_size:.2f}MB " +
                      f"({(1 - optimized_size/original_size)*100:.1f}% reduction)")
            
            # Save as CSV for ML algorithms
            csv_file = os.path.join(self.config.cache_base_dir, 'ml_ready', 'station_features.csv')
            optimized_df.to_csv(csv_file, index=False)
            
            # Also save as compressed parquet for more efficient storage/retrieval
            parquet_file = os.path.join(self.config.cache_base_dir, 'ml_ready', 'station_features.parquet')
            optimized_df.to_parquet(parquet_file, compression='snappy')
            
            # Save feature metadata
            metadata = {
                'total_stations': len(optimized_df),
                'total_features': len(optimized_df.columns) - 1,  # Exclude station_id
                'feature_categories': self._analyze_feature_categories(optimized_df.columns),
                'memory_usage_mb': {
                    'original': original_size,
                    'optimized': optimized_size,
                    'reduction_percent': (1 - optimized_size/original_size)*100
                },
                'creation_time': datetime.now().isoformat(),
                'config': asdict(self.config),
                'missing_values': {
                    'original_count': features_df.isna().sum().sum(),
                    'remaining_count': optimized_df.isna().sum().sum()
                }
            }
            
            metadata_file = os.path.join(self.config.cache_base_dir, 'ml_ready', 'feature_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ ML-ready features created: {len(optimized_df)} stations, {len(optimized_df.columns)-1} features")
            logger.info(f"📁 Files saved:")
            logger.info(f"   - Features CSV: {csv_file}")
            logger.info(f"   - Features Parquet: {parquet_file}")
            logger.info(f"   - Metadata: {metadata_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ML-ready feature creation failed: {e}")
            return False
    
    def _analyze_feature_categories(self, columns: List[str]) -> Dict[str, int]:
        """Analyze feature categories for metadata"""
        categories = {
            'traditional_osm': 0,
            'population': 0,
            'comprehensive_osm': 0,
            'coordinates': 0,
            'other': 0
        }
        
        for col in columns:
            if col.startswith('trad_'):
                categories['traditional_osm'] += 1
            elif col.startswith('pop_'):
                categories['population'] += 1
            elif col.startswith('comp_'):
                categories['comprehensive_osm'] += 1
            elif col in ['lat', 'lon', 'station_id']:
                categories['coordinates'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _save_cache(self, cache_file: str, data: Dict):
        """Save data to cache file with error handling and compression for large files"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Convert all keys to strings for JSON serialization
            cache_data = {str(k): v for k, v in data.items()}
            
            # Determine if we should compress
            if len(json.dumps(cache_data)) > 10 * 1024 * 1024:  # >10MB
                import gzip
                with gzip.open(cache_file + '.gz', 'wt', encoding='utf-8') as f:
                    json.dump(cache_data, f)
                logger.info(f"💾 Saved compressed cache: {cache_file}.gz")
            else:
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.info(f"💾 Saved cache: {cache_file}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save cache {cache_file}: {e}")
    
    def _generate_download_summary(self):
        """Generate comprehensive download summary with performance metrics"""
        logger.info("📋 Generating download summary...")
        
        duration = 0
        if self.download_stats['start_time'] and self.download_stats['end_time']:
            duration = self.download_stats['end_time'] - self.download_stats['start_time']
        
        # Get cache statistics
        cache_stats = self.cache_manager.get_stats()
        
        # Get rate limiter statistics
        rate_limiter_stats = {k: v.get_stats() for k, v in self.rate_limiters.items()}
        
        # Calculate key performance metrics
        stations_per_second = self.download_stats['total_stations'] / max(1, duration)
        api_efficiency = 1 - (len(self.download_stats['errors']) / max(1, self.download_stats['total_stations']))
        
        summary = {
            'download_summary': {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'total_stations': self.download_stats['total_stations'],
                'successful_stations': self.download_stats['successful_stations'],
                'failed_stations': self.download_stats['failed_stations'],
                'success_rate': (self.download_stats['successful_stations'] / 
                               max(1, self.download_stats['total_stations'])),
                'errors_count': len(self.download_stats['errors']),
                'recent_errors': self.download_stats['errors'][-10:]  # Last 10 errors
            },
            'performance_metrics': {
                'stations_per_second': stations_per_second,
                'cache_hit_rate': cache_stats['hit_rate'],
                'api_efficiency': api_efficiency,
                'memory_usage_mb': cache_stats['memory_usage_mb'],
                'rate_limiter_stats': rate_limiter_stats
            },
            'cache_status': self._get_cache_status(),
            'config': asdict(self.config)
        }
        
        summary_file = os.path.join(self.config.cache_base_dir, 'download_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("🎯 DOWNLOAD SUMMARY")
        logger.info("=" * 80)
        logger.info(f"⏱️  Duration: {duration:.1f} seconds")
        logger.info(f"🏢 Total stations: {self.download_stats['total_stations']}")
        logger.info(f"✅ Successful: {self.download_stats['successful_stations']}")
        logger.info(f"❌ Failed: {self.download_stats['failed_stations']}")
        logger.info(f"📊 Success rate: {summary['download_summary']['success_rate']*100:.1f}%")
        logger.info(f"🚀 Performance: {stations_per_second:.2f} stations/second")
        logger.info(f"💾 Cache hit rate: {cache_stats['hit_rate']*100:.1f}%")
        logger.info(f"🔄 API efficiency: {api_efficiency*100:.1f}%")
        logger.info(f"💾 Summary saved: {summary_file}")
        logger.info("=" * 80)
    
    def _get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status"""
        cache_files = {
            'traditional_osm': os.path.join(self.config.cache_base_dir, 'consolidated', 'traditional_osm_features.json'),
            'population': os.path.join(self.config.cache_base_dir, 'consolidated', 'population_features.json'),
            'comprehensive_osm': os.path.join(self.config.cache_base_dir, 'consolidated', 'comprehensive_osm_features.json'),
            'ml_ready_json': os.path.join(self.config.cache_base_dir, 'ml_ready', 'all_features.json'),
            'ml_ready_csv': os.path.join(self.config.cache_base_dir, 'ml_ready', 'station_features.csv'),
            'ml_ready_parquet': os.path.join(self.config.cache_base_dir, 'ml_ready', 'station_features.parquet')
        }
        
        status = {}
        for cache_type, cache_file in cache_files.items():
            # Check both compressed and uncompressed
            exists = os.path.exists(cache_file) or os.path.exists(cache_file + '.gz')
            
            if exists:
                if os.path.exists(cache_file):
                    size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                    path = cache_file
                else:
                    size_mb = os.path.getsize(cache_file + '.gz') / (1024 * 1024)
                    path = cache_file + '.gz'
                
                status[cache_type] = {
                    'exists': True,
                    'size_mb': round(size_mb, 2),
                    'path': path
                }
            else:
                status[cache_type] = {
                    'exists': False,
                    'size_mb': 0,
                    'path': cache_file
                }
        
        return status

def main():
    """Command line interface for the master downloader"""
    parser = argparse.ArgumentParser(description="Optimized OSM Cache Downloader for ML Algorithms")
    
    # Input/Output
    parser.add_argument("--trips-file", default="data/trips_8days_flat.csv", 
                       help="Path to trips CSV file")
    parser.add_argument("--cache-dir", default="cache", 
                       help="Base cache directory")
    
    # Download options
    parser.add_argument("--radius", type=int, default=1000, 
                       help="Radius in meters for feature extraction")
    parser.add_argument("--no-traditional", action="store_true", 
                       help="Skip traditional OSM download")
    parser.add_argument("--no-osmnx", action="store_true", 
                       help="Skip OSMnx comprehensive download")
    parser.add_argument("--no-population", action="store_true", 
                       help="Skip population features download")
    
    # Performance
    parser.add_argument("--batch-size", type=int, default=20, 
                       help="Batch size for processing")
    parser.add_argument("--rate-limit", type=float, default=1.0, 
                       help="Initial rate limit in seconds between requests")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="Maximum worker threads (lower is gentler on APIs)")
    parser.add_argument("--memory-cache", type=int, default=512, 
                       help="Memory cache size limit in MB")
    parser.add_argument("--no-adaptive-rate", action="store_true", 
                       help="Disable adaptive rate limiting")
    parser.add_argument("--gentle-mode", action="store_true",
                       help="Enable extra gentle mode for heavily rate-limited APIs")
    
    # Control options
    parser.add_argument("--force-refresh", action="store_true", 
                       help="Force refresh existing caches")
    parser.add_argument("--no-consolidate", action="store_true", 
                       help="Skip cache consolidation")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Cleanup individual cache files after consolidation")
    
    # Actions
    parser.add_argument("--status", action="store_true", 
                       help="Show cache status only")
    
    args = parser.parse_args()
    
    # Create configuration
    config = CacheConfig(
        cache_base_dir=args.cache_dir,
        trips_file=args.trips_file,
        radius_meters=args.radius,
        use_traditional_osm=not args.no_traditional,
        use_osmnx=not args.no_osmnx,
        use_population=not args.no_population,
        batch_size=args.batch_size,
        rate_limit_seconds=args.rate_limit,
        max_workers=args.max_workers,
        force_refresh=args.force_refresh,
        consolidate_cache=not args.no_consolidate,
        cleanup_individual_files=args.cleanup,
        memory_cache_limit_mb=args.memory_cache,
        use_adaptive_rate_limiting=not args.no_adaptive_rate,
        gentle_mode=args.gentle_mode
    )
    
    # Initialize downloader
    downloader = MasterOSMCacheDownloader(config)
    
    # Show status if requested
    if args.status:
        cache_status = downloader._get_cache_status()
        print("\n📊 CACHE STATUS:")
        print("=" * 50)
        for cache_type, status in cache_status.items():
            status_icon = "✅" if status['exists'] else "❌"
            print(f"{status_icon} {cache_type}: {status['size_mb']} MB")
            if status['exists']:
                print(f"   📁 {status['path']}")
        return
    
    # Run the complete download process
    logger.info("🚀 Starting Optimized OSM Cache Download Process")
    
    # Step 1: Load station coordinates
    if not downloader.load_station_coordinates():
        logger.error("❌ Failed to load station coordinates. Exiting.")
        sys.exit(1)
    
    # Step 2: Download all features
    if not downloader.download_all_features():
        logger.error("❌ Feature download failed. Check logs for details.")
        sys.exit(1)
    
    logger.info("🎉 Optimized OSM Cache Download completed successfully!")
    logger.info("💡 Your ML-ready features are available in:")
    logger.info(f"   📄 CSV: {os.path.join(config.cache_base_dir, 'ml_ready', 'station_features.csv')}")
    logger.info(f"   📦 Parquet: {os.path.join(config.cache_base_dir, 'ml_ready', 'station_features.parquet')}")
    logger.info(f"   📋 JSON: {os.path.join(config.cache_base_dir, 'ml_ready', 'all_features.json')}")

if __name__ == "__main__":
    main()
