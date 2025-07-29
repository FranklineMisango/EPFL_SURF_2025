"""
Demo script for the Enhanced Bike Flow Prediction System
Demonstrates OSM feature extraction, multi-path routing, and ML evaluation
"""

import pandas as pd
import numpy as np
from osm_feature_extractor import OSMFeatureExtractor, FeatureInfluenceAnalyzer
from multi_path_router import MultiPathRouter
from ml_evaluation_system import MLEvaluationSystem
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import time

def demo_osm_feature_extraction():
    """Demonstrate OSM feature extraction"""
    print("=" * 60)
    print("🗺️  DEMO: OpenStreetMap Feature Extraction")
    print("=" * 60)
    
    # Initialize extractor
    extractor = OSMFeatureExtractor()
    
    # Example coordinates (Lausanne, Switzerland - EPFL area)
    test_locations = [
        (46.5197, 6.6323, "EPFL Campus"),
        (46.5238, 6.6356, "Lausanne Center"),
        (46.5167, 6.6407, "Ouchy Lakefront")
    ]
    
    for lat, lon, name in test_locations:
        print(f"\n📍 Extracting features around {name} ({lat:.4f}, {lon:.4f})")
        print("-" * 50)
        
        try:
            # Extract features
            features = extractor.extract_features_around_station(lat, lon, radius_m=500)
            
            # Show summary
            print("Feature Summary:")
            for feature_type, feature_list in features.items():
                if len(feature_list) > 0:
                    print(f"  {feature_type}: {len(feature_list)} found")
            
            # Compute metrics
            metrics = extractor.compute_feature_metrics(features)
            
            print("\nTop Metrics:")
            sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
            for key, value in sorted_metrics[:8]:
                if value > 0:
                    print(f"  {key}: {value}")
            
            # Rate limiting for demo
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Feature descriptions
    print(f"\n📋 Available Feature Types:")
    descriptions = extractor.get_feature_descriptions()
    for i, (feature_type, description) in enumerate(list(descriptions.items())[:10], 1):
        print(f"  {i:2d}. {feature_type}: {description}")
    
    print(f"\n✅ OSM Feature Extraction Demo Complete!")

def demo_multi_path_routing():
    """Demonstrate multi-path routing"""
    print("\n" + "=" * 60)
    print("🛣️  DEMO: Multi-Path Routing")
    print("=" * 60)
    
    # Initialize router
    router = MultiPathRouter()
    
    # Example route (Lausanne area)
    start_lat, start_lon = 46.5197, 6.6323  # EPFL
    end_lat, end_lon = 46.5238, 6.6356      # Lausanne Center
    
    print(f"📍 Route: EPFL → Lausanne Center")
    print(f"   Start: ({start_lat:.4f}, {start_lon:.4f})")
    print(f"   End:   ({end_lat:.4f}, {end_lon:.4f})")
    print("-" * 50)
    
    try:
        # Get multiple paths
        print("🔍 Generating multiple path options...")
        paths = router.get_multiple_paths(start_lat, start_lon, end_lat, end_lon, max_paths=5)
        
        print(f"\n✅ Generated {len(paths)} path options:")
        
        for i, path in enumerate(paths, 1):
            print(f"\n{i}. {path.path_type.upper()} ({path.source})")
            print(f"   Distance: {path.distance_m/1000:.2f} km")
            print(f"   Duration: {path.duration_s/60:.1f} minutes")
            print(f"   Profile:  {path.routing_profile}")
            print(f"   Confidence: {path.confidence:.2f}")
            print(f"   Coordinates: {len(path.coordinates)} points")
        
        # Analyze path diversity
        print(f"\n📊 Path Diversity Analysis:")
        diversity = router.analyze_path_diversity(paths)
        for key, value in diversity.items():
            print(f"   {key}: {value:.3f}")
        
        # Path summary table
        print(f"\n📋 Path Summary Table:")
        summary_df = router.get_path_summary(paths)
        print(summary_df.to_string(index=False, float_format='%.2f'))
        
    except Exception as e:
        print(f"❌ Error in multi-path routing: {e}")
    
    print(f"\n✅ Multi-Path Routing Demo Complete!")

def demo_ml_evaluation():
    """Demonstrate ML evaluation system"""
    print("\n" + "=" * 60)
    print("🎯 DEMO: ML Evaluation System")
    print("=" * 60)
    
    # Initialize evaluation system
    evaluator = MLEvaluationSystem()
    
    # Generate sample data (simulating bike flow prediction)
    print("🔧 Generating sample bike flow data...")
    X, y = make_regression(
        n_samples=1000, 
        n_features=25, 
        noise=0.1, 
        random_state=42
    )
    
    # Create feature names (simulating bike flow features)
    feature_names = [
        'hour', 'day_of_week', 'is_weekend', 'temperature',
        'start_lat', 'start_lon', 'end_lat', 'end_lon',
        'distance', 'historical_flow', 'start_popularity', 'end_popularity'
    ]
    
    # Add OSM features
    osm_features = [
        'hotels_count', 'restaurants_count', 'banks_count', 'shops_count',
        'schools_count', 'parks_count', 'offices_count', 'residential_count',
        'bus_stops_count', 'bike_lanes_length', 'parking_count', 'cafes_count', 'hospitals_count'
    ]
    
    feature_names.extend(osm_features)
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🏷️  Features include: temporal, spatial, historical, and OSM features")
    
    # Create and train model
    print("\n🤖 Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Comprehensive evaluation
    print("🔍 Running comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(model, X, y, feature_names, cv_folds=5)
    
    # Display results
    print("\n📈 EVALUATION RESULTS:")
    print("-" * 30)
    
    # Summary
    summary = results['summary']
    print(f"Overall Score: {summary['overall_score']:.1f}/100")
    print(f"Confidence Grade: {summary['confidence_grade']}")
    
    # Cross-validation
    cv_results = results['cross_validation']
    print(f"\n📊 Cross-Validation (5-fold):")
    for metric, values in cv_results.items():
        mean_val = values['mean']
        std_val = values['std']
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Hold-out results
    holdout = results['holdout']
    print(f"\n🎯 Hold-out Test Results:")
    for metric, value in holdout.items():
        if isinstance(value, (int, float)):
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Feature importance
    importance = results.get('feature_importance', {})
    if 'top_features' in importance:
        print(f"\n🏆 Top 10 Most Important Features:")
        for i, feature in enumerate(importance['top_features'][:10], 1):
            print(f"  {i:2d}. {feature}")
    
    # Confidence analysis
    confidence = results.get('confidence_analysis', {})
    if confidence:
        coverage = confidence.get('coverage', {})
        print(f"\n🎯 Confidence Analysis:")
        print(f"  95% Prediction Interval Coverage: {coverage.get('coverage_95', 0):.1%}")
        print(f"  90% Prediction Interval Coverage: {coverage.get('coverage_90', 0):.1%}")
        
        uncertainty = confidence.get('uncertainty_stats', {})
        print(f"  Mean Uncertainty: {uncertainty.get('mean_uncertainty', 0):.4f}")
    
    # Model stability
    stability = results.get('stability_analysis', {})
    if stability:
        print(f"\n⚖️  Model Stability:")
        for metric, stats in stability.items():
            if isinstance(stats, dict):
                cv_coeff = stats.get('cv', 0)  # Coefficient of variation
                print(f"  {metric.upper()} CV: {cv_coeff:.3f} (lower is better)")
    
    # Generate detailed report
    print(f"\n📋 Generating detailed evaluation report...")
    report = evaluator.generate_evaluation_report(results)
    
    print("\n" + "="*60)
    print("📄 DETAILED EVALUATION REPORT")
    print("="*60)
    print(report)
    
    # Demonstrate confidence predictions
    print(f"\n🔮 Testing Confidence Predictions...")
    
    # Train model for confidence prediction
    model.fit(X[:800], y[:800])
    
    # Get confidence predictions for test samples
    test_X = X[800:810]
    confidence_preds = evaluator.predict_with_confidence(model, test_X, confidence_level=0.95)
    
    print(f"\nSample Confidence Predictions (95% confidence):")
    for i, conf in enumerate(confidence_preds[:5]):
        print(f"  Sample {i+1}: {conf.prediction:.2f} [{conf.lower_bound:.2f}, {conf.upper_bound:.2f}]")
        print(f"             Uncertainty: ±{conf.uncertainty:.2f}, Width: {conf.prediction_interval_width:.2f}")
    
    print(f"\n✅ ML Evaluation Demo Complete!")

def demo_feature_influence_analysis():
    """Demonstrate feature influence analysis"""
    print("\n" + "=" * 60)
    print("📊 DEMO: Feature Influence Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = FeatureInfluenceAnalyzer()
    
    # Create sample station features (simulating real data)
    print("🔧 Creating sample station features with OSM data...")
    
    station_data = []
    for i in range(50):  # 50 sample stations
        station_data.append({
            'station_id': i,
            'lat': 46.5 + np.random.normal(0, 0.02),
            'lon': 6.6 + np.random.normal(0, 0.02),
            'total_trips': np.random.poisson(1000),
            'hotels_count': np.random.poisson(2),
            'restaurants_count': np.random.poisson(5),
            'banks_count': np.random.poisson(1),
            'shops_count': np.random.poisson(8),
            'schools_count': np.random.poisson(1),
            'parks_count': np.random.poisson(2),
            'offices_count': np.random.poisson(10),
            'residential_count': np.random.poisson(15),
            'bus_stops_count': np.random.poisson(3),
            'parking_count': np.random.poisson(2)
        })
    
    station_features_df = pd.DataFrame(station_data)
    
    # Create sample flow data
    flow_data = []
    for i in range(200):  # 200 sample flows
        destination = np.random.choice(station_features_df['station_id'])
        
        # Simulate flow influenced by features
        dest_features = station_features_df[station_features_df['station_id'] == destination].iloc[0]
        
        # Flow influenced by restaurants, offices, and schools
        base_flow = (
            dest_features['restaurants_count'] * 2 +
            dest_features['offices_count'] * 1.5 +
            dest_features['schools_count'] * 3 +
            np.random.normal(0, 2)
        )
        
        flow_data.append({
            'destination_station': destination,
            'flow_volume': max(1, base_flow)
        })
    
    flow_df = pd.DataFrame(flow_data)
    
    print(f"📊 Analyzing influence of {len(station_features_df.columns)-1} features on bike flows...")
    
    # Analyze feature influence
    influence_scores = analyzer.analyze_feature_influence(station_features_df, flow_df)
    
    print(f"\n🏆 Feature Influence Rankings:")
    print("-" * 40)
    
    for i, (feature, score) in enumerate(list(influence_scores.items())[:15], 1):
        print(f"{i:2d}. {feature:20s}: {score:.4f}")
    
    # Get top influential features
    top_features = analyzer.get_top_influential_features(10)
    
    print(f"\n🎯 Top 10 Most Influential Features:")
    for i, (feature, score) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feature}: {score:.4f}")
    
    # Create influence report
    print(f"\n📋 Generating influence analysis report...")
    
    # Mock extractor for report generation
    class MockExtractor:
        def get_feature_descriptions(self):
            return {
                'hotels': 'Hotels and accommodations',
                'restaurants': 'Restaurants and dining establishments',
                'banks': 'Banks and financial services',
                'shops': 'Retail shops and stores',
                'schools': 'Educational institutions',
                'parks': 'Parks and green spaces',
                'offices': 'Office buildings and workplaces',
                'residential': 'Residential areas',
                'bus_stops': 'Public transport stops',
                'parking': 'Parking facilities'
            }
        
        def get_feature_influences(self):
            return {
                'hotels': 'Tourist destinations and temporary stays',
                'restaurants': 'Meal destinations and social gatherings',
                'banks': 'Financial errands and business trips',
                'shops': 'Shopping trips and daily errands',
                'schools': 'Student commutes and family trips',
                'parks': 'Recreation and leisure activities',
                'offices': 'Work commutes and business meetings',
                'residential': 'Home origins and destinations',
                'bus_stops': 'Multimodal transport connections',
                'parking': 'Car-bike transfer points'
            }
    
    mock_extractor = MockExtractor()
    report = analyzer.create_influence_report(mock_extractor)
    
    print("\n" + "="*60)
    print("📄 FEATURE INFLUENCE REPORT")
    print("="*60)
    print(report)
    
    print(f"\n✅ Feature Influence Analysis Demo Complete!")

def main():
    """Run all demos"""
    print("🚴‍♂️ ENHANCED BIKE FLOW PREDICTION SYSTEM DEMO")
    print("=" * 80)
    print("This demo showcases the enhanced features:")
    print("1. 🗺️  OpenStreetMap feature extraction")
    print("2. 🛣️  Multi-path routing with alternatives")
    print("3. 🎯 Robust ML evaluation with confidence metrics")
    print("4. 📊 Feature influence analysis")
    print("=" * 80)
    
    try:
        # Run demos
        demo_osm_feature_extraction()
        demo_multi_path_routing()
        demo_ml_evaluation()
        demo_feature_influence_analysis()
        
        print("\n" + "=" * 80)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Run 'streamlit run enhanced_bike_flow_app.py' to see the full system")
        print("2. Explore the interactive map with OSM features")
        print("3. Test different stations and time periods")
        print("4. Review the ML evaluation metrics and confidence scores")
        print("5. Analyze feature influence on destination predictions")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your internet connection and dependencies.")

if __name__ == "__main__":
    main()