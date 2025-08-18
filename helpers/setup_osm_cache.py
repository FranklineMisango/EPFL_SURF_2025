#!/usr/bin/env python3
"""
Quick Start Guide for OSM Cache Download and ML Preparation
Run this before your ML algorithms to ensure all OSM features are cached and ready.
"""

import os
import sys
import pandas as pd
import logging
from master_osm_cache_downloader import MasterOSMCacheDownloader, CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_cache_download(station_ids=None, feature_types=None, cache_prefix=""):
    """Quick setup and download of OSM features for ML
    
    Args:
        station_ids: List of specific station IDs to download (default: all stations)
        feature_types: List of feature types ['traditional', 'population', 'comprehensive'] 
        cache_prefix: Prefix for cache files (e.g., "lausanne_")
    """
    
    print("🚀 OSM Feature Cache Setup for ML Algorithms")
    print("=" * 60)
    
    # Check if trips data exists
    trips_file = "data/trips_8days_flat.csv"
    if not os.path.exists(trips_file):
        logger.error(f"❌ Trips file not found: {trips_file}")
        logger.info("💡 Make sure your trips data is in the correct location.")
        return False
    
    # Set feature types based on input
    if feature_types is None:
        feature_types = ['traditional', 'population', 'comprehensive']
    
    # Configure for fast ML preparation
    config = CacheConfig(
        trips_file=trips_file,
        cache_base_dir=f"cache/{cache_prefix.rstrip('_')}" if cache_prefix else "cache",
        radius_meters=1000,        # 1km radius around each station
        use_traditional_osm='traditional' in feature_types,
        use_osmnx='comprehensive' in feature_types,
        use_population='population' in feature_types,
        batch_size=10,            # Process 10 stations at a time
        rate_limit_seconds=0.5,   # 0.5 second delay between requests
        force_refresh=False,      # Use existing cache if available
        consolidate_cache=True,   # Create unified ML-ready cache
        create_summary_stats=True # Generate download statistics
    )
    
    # Initialize downloader
    downloader = MasterOSMCacheDownloader(config)
    
    # Load station coordinates
    logger.info("📍 Loading station coordinates...")
    if not downloader.load_station_coordinates():
        logger.error("❌ Failed to load station coordinates")
        return False
    
    # Filter to specific stations if provided
    if station_ids is not None:
        original_count = len(downloader.station_coords)
        downloader.station_coords = {
            sid: coords for sid, coords in downloader.station_coords.items() 
            if sid in station_ids
        }
        logger.info(f"📍 Filtered to {len(downloader.station_coords)} stations (from {original_count})")
    
    logger.info(f"🏢 Processing {len(downloader.station_coords)} stations")
    
    if len(downloader.station_coords) == 0:
        logger.error("❌ No stations to process")
        return False
    
    # Download all features
    logger.info("⬇️ Downloading OSM features (this may take a while)...")
    if not downloader.download_all_features():
        logger.error("❌ Feature download failed")
        return False
    
    logger.info("✅ OSM feature cache setup completed!")
    return True

def load_cached_features(cache_file=None):
    """Load cached features for ML algorithms
    
    Args:
        cache_file: Path to cached features CSV file. If None, tries to auto-detect.
    """
    
    # If no specific cache file provided, try to find one
    if cache_file is None:
        # Try different possible locations
        possible_locations = [
            "cache/ml_ready/station_features.csv",  # Original location
            "cache/lausanne/ml_ready/station_features.csv",  # Lausanne cache
            "cache/bern/ml_ready/station_features.csv",  # Bern cache
            "cache/zurich/ml_ready/station_features.csv",  # Zurich cache
        ]
        
        cache_file = None
        for location in possible_locations:
            if os.path.exists(location):
                cache_file = location
                logger.info(f"🔍 Auto-detected cache: {cache_file}")
                break
        
        if cache_file is None:
            logger.error(f"❌ No cached features found in any of these locations:")
            for location in possible_locations:
                logger.error(f"   - {location}")
            logger.info("💡 Run location-based downloader first:")
            logger.info("   python location_based_osm_downloader.py --city lausanne --max-stations 5")
            return None
    
    if not os.path.exists(cache_file):
        logger.error(f"❌ Cached features not found: {cache_file}")
        logger.info("💡 Run quick_cache_download() or location-based downloader first.")
        return None
    
    try:
        features_df = pd.read_csv(cache_file)
        logger.info(f"✅ Loaded cached features: {len(features_df)} stations, {len(features_df.columns)-1} features")
        
        # Show feature categories
        feature_categories = {
            'Traditional OSM': len([c for c in features_df.columns if c.startswith('trad_')]),
            'Population': len([c for c in features_df.columns if c.startswith('pop_')]),
            'Comprehensive OSM': len([c for c in features_df.columns if c.startswith('comp_')]),
            'Coordinates': len([c for c in features_df.columns if c in ['lat', 'lon', 'station_id']])
        }
        
        logger.info("📊 Feature categories:")
        for category, count in feature_categories.items():
            logger.info(f"   {category}: {count} features")
        
        return features_df
        
    except Exception as e:
        logger.error(f"❌ Failed to load cached features: {e}")
        return None

def load_city_cached_features(city):
    """Load cached features for a specific city
    
    Args:
        city: City name (e.g., 'lausanne', 'bern', 'zurich')
        
    Returns:
        DataFrame with cached OSM features for the city, or None if not found
    """
    
    cache_file = f"cache/{city.lower()}/ml_ready/station_features.csv"
    
    if not os.path.exists(cache_file):
        logger.error(f"❌ No cached features found for {city}: {cache_file}")
        logger.info(f"💡 Download {city} features first:")
        logger.info(f"   python location_based_osm_downloader.py --city {city.lower()} --max-stations 10")
        return None
    
    try:
        features_df = pd.read_csv(cache_file)
        logger.info(f"✅ Loaded {city} cached features: {len(features_df)} stations, {len(features_df.columns)-1} features")
        return features_df
        
    except Exception as e:
        logger.error(f"❌ Failed to load {city} cached features: {e}")
        return None

def get_available_city_caches():
    """Get list of cities that have cached OSM features
    
    Returns:
        List of city names that have cached features available
    """
    
    available_cities = []
    
    # Check common city cache directories
    cache_base = "cache"
    if os.path.exists(cache_base):
        for item in os.listdir(cache_base):
            city_cache_dir = os.path.join(cache_base, item)
            if os.path.isdir(city_cache_dir):
                features_file = os.path.join(city_cache_dir, "ml_ready", "station_features.csv")
                if os.path.exists(features_file):
                    available_cities.append(item)
    
    return available_cities

def check_cache_status(station_ids=None):
    """Check the status of all caches
    
    Args:
        station_ids: List of specific station IDs to check (default: all stations)
    """
    
    print("\n📊 OSM Cache Status Report")
    print("=" * 50)
    
    config = CacheConfig()
    downloader = MasterOSMCacheDownloader(config)
    
    # Filter to specific stations if provided
    if station_ids is not None:
        downloader.load_station_coordinates()
        downloader.station_coords = {
            sid: coords for sid, coords in downloader.station_coords.items() 
            if sid in station_ids
        }
        print(f"📍 Checking cache for {len(downloader.station_coords)} stations")
    
    cache_status = downloader._get_cache_status()
    
    for cache_type, status in cache_status.items():
        status_icon = "✅" if status['exists'] else "❌"
        cache_name = cache_type.replace('_', ' ').title()
        
        if status['exists']:
            print(f"{status_icon} {cache_name}: {status['size_mb']:.1f} MB")
            if 'path' in status:
                print(f"    📁 Path: {status['path']}")
        else:
            print(f"{status_icon} {cache_name}: Not available")
    
    # Check if ready for ML
    ml_ready = cache_status.get('ml_ready_csv', {}).get('exists', False)
    
    print("\n🤖 ML Readiness:")
    if ml_ready:
        print("✅ Ready for ML algorithms!")
        if 'path' in cache_status.get('ml_ready_csv', {}):
            print(f"📁 Features: {cache_status['ml_ready_csv']['path']}")
    else:
        print("❌ Not ready for ML algorithms")
        print("💡 Run quick_cache_download() to prepare features")
    
    return cache_status

def main():
    """Main function for command line usage"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "download":
            success = quick_cache_download()
            sys.exit(0 if success else 1)
            
        elif command == "status":
            check_cache_status()
            
        elif command == "load":
            features_df = load_cached_features()
            if features_df is not None:
                print(f"\n📋 Feature Preview:")
                print(features_df.head())
            
        else:
            print("❌ Unknown command. Use: download, status, or load")
            sys.exit(1)
    else:
        # Interactive mode
        print("🚀 OSM Cache Interactive Setup")
        print("\nOptions:")
        print("1. Download OSM features")
        print("2. Check cache status") 
        print("3. Load cached features")
        print("4. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (1-4): ").strip()
                
                if choice == "1":
                    quick_cache_download()
                elif choice == "2":
                    check_cache_status()
                elif choice == "3":
                    features_df = load_cached_features()
                    if features_df is not None:
                        print(f"\n📋 Feature Preview (first 5 rows):")
                        print(features_df.head())
                elif choice == "4":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
