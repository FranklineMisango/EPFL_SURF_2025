"""
Enhanced New Station Predictor with Map Integration and Accuracy Testing
Works with existing OSM map loader and provides comprehensive evaluation.
"""

import numpy as np
import pandas as pd
import torch
import logging
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from new_station_predictor import NewStationPredictor
from gnn_flow_predictor import GNNFlowPredictor
from optimized_map import create_optimized_map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMapIntegratedPredictor:
    """Enhanced predictor that integrates with existing OSM map and provides accuracy evaluation"""
    
    def __init__(self, trained_predictor: GNNFlowPredictor):
        """Initialize with trained GNN predictor"""
        self.trained_predictor = trained_predictor
        self.new_station_predictor = NewStationPredictor(trained_predictor)
        self.test_results = {}
        
    def add_new_stations_from_coordinates(self, station_coords: Dict[str, Tuple[float, float]], 
                                        station_names: Dict[str, str] = None) -> Dict[str, bool]:
        """Add multiple new stations from coordinate dictionary"""
        results = {}
        
        for station_id, (lat, lon) in station_coords.items():
            name = station_names.get(station_id) if station_names else None
            success = self.new_station_predictor.add_new_station(station_id, lat, lon, name)
            results[station_id] = success
            
        logger.info(f"Added {sum(results.values())}/{len(results)} new stations successfully")
        return results
    
    def evaluate_new_station_accuracy(self, test_station_id: str) -> Dict[str, Any]:
        """Evaluate prediction accuracy for a new station using held-out data"""
        
        if test_station_id not in self.trained_predictor.station_coords:
            logger.error(f"Test station {test_station_id} not found in training data")
            return {'error': 'Station not found'}
        
        # Remove station from trained model temporarily for testing
        original_coords = self.trained_predictor.station_coords[test_station_id]
        original_features = self.trained_predictor.station_features[test_station_id]
        
        try:
            # Temporarily remove from training data
            del self.trained_predictor.station_coords[test_station_id]
            del self.trained_predictor.station_features[test_station_id]
            
            # Add as new station to predictor
            lat, lon = original_coords
            self.new_station_predictor.add_new_station(test_station_id, lat, lon)
            
            # Get all other stations for prediction
            other_stations = list(self.trained_predictor.station_coords.keys())
            
            # Extract actual trips for this station from original data
            test_trips = self.trained_predictor.trips_df[
                (self.trained_predictor.trips_df['start_station_id'] == test_station_id) |
                (self.trained_predictor.trips_df['end_station_id'] == test_station_id)
            ]
            
            # Generate predictions for various time periods
            predictions_data = []
            actual_data = []
            
            # Sample test periods
            test_periods = [(h, d) for h in range(0, 24, 4) for d in range(7)]  # Every 4 hours, all days
            
            for hour, dow in test_periods:
                # Predict outflows
                predicted_outflows = self.new_station_predictor.predict_flows_from_new_station(
                    test_station_id, other_stations, hour, dow
                )
                
                # Predict inflows
                predicted_inflows = self.new_station_predictor.predict_flows_to_new_station(
                    other_stations, test_station_id, hour, dow
                )
                
                # Get actual flows for this time period
                actual_trips_period = test_trips[
                    (test_trips['hour'] == hour) & (test_trips['day_of_week'] == dow)
                ]
                
                # Calculate actual outflows
                actual_outflows = actual_trips_period[
                    actual_trips_period['start_station_id'] == test_station_id
                ].groupby('end_station_id').size().to_dict()
                
                # Calculate actual inflows
                actual_inflows = actual_trips_period[
                    actual_trips_period['end_station_id'] == test_station_id
                ].groupby('start_station_id').size().to_dict()
                
                # Compare predictions vs actual for each station pair
                for station in other_stations:
                    # Outflow comparison
                    pred_out = predicted_outflows.get(station, 0)
                    actual_out = actual_outflows.get(station, 0)
                    
                    predictions_data.append(pred_out)
                    actual_data.append(actual_out)
                    
                    # Inflow comparison
                    pred_in = predicted_inflows.get(station, 0)
                    actual_in = actual_inflows.get(station, 0)
                    
                    predictions_data.append(pred_in)
                    actual_data.append(actual_in)
            
            # Calculate accuracy metrics
            predictions_array = np.array(predictions_data)
            actual_array = np.array(actual_data)
            
            # Remove zero-zero pairs to avoid skewing metrics
            non_zero_mask = (predictions_array != 0) | (actual_array != 0)
            pred_filtered = predictions_array[non_zero_mask]
            actual_filtered = actual_array[non_zero_mask]
            
            if len(pred_filtered) > 0:
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(actual_filtered, pred_filtered)),
                    'mae': mean_absolute_error(actual_filtered, pred_filtered),
                    'r2_score': r2_score(actual_filtered, pred_filtered) if len(set(actual_filtered)) > 1 else 0,
                    'mean_actual': np.mean(actual_filtered),
                    'mean_predicted': np.mean(pred_filtered),
                    'correlation': np.corrcoef(actual_filtered, pred_filtered)[0, 1] if len(pred_filtered) > 1 else 0,
                    'n_comparisons': len(pred_filtered)
                }
            else:
                metrics = {'error': 'No valid comparisons found'}
            
            self.test_results[test_station_id] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating accuracy for {test_station_id}: {e}")
            return {'error': str(e)}
            
        finally:
            # Restore original station data
            self.trained_predictor.station_coords[test_station_id] = original_coords
            self.trained_predictor.station_features[test_station_id] = original_features
    
    def create_enhanced_osm_map(self, new_station_ids: List[str] = None, 
                               hour: int = 8, day_of_week: int = 1,
                               show_predictions: bool = True,
                               show_osm_features: bool = True) -> folium.Map:
        """Create enhanced OSM map with new station predictions and features"""
        
        # Start with base optimized map
        m = create_optimized_map(self.trained_predictor, hour, day_of_week)
        
        if new_station_ids:
            self._add_new_station_layers(m, new_station_ids, hour, day_of_week, 
                                       show_predictions, show_osm_features)
        
        if show_osm_features:
            self._add_osm_feature_overlays(m)
        
        return m
    
    def _add_new_station_layers(self, m: folium.Map, new_station_ids: List[str], 
                               hour: int, day_of_week: int, show_predictions: bool, 
                               show_osm_features: bool):
        """Add new station markers and predictions to map"""
        
        # Create new station group
        new_station_group = folium.FeatureGroup(
            name="New Stations", 
            overlay=True,
            control=True
        )
        
        for station_id in new_station_ids:
            if station_id not in self.new_station_predictor.new_stations:
                continue
                
            station = self.new_station_predictor.new_stations[station_id]
            lat, lon = station['lat'], station['lon']
            name = station['name']
            
            # Get OSM features for popup
            features = self.new_station_predictor.new_station_features.get(station_id, {})
            osm_features = {k.replace('cached_', '').replace('osm_', ''): v 
                          for k, v in features.items() 
                          if k.startswith(('cached_', 'osm_')) and v > 0}
            
            # Create popup content
            popup_content = f"""
            <b>New {name}</b><br>
            <b>Station ID:</b> {station_id}<br>
            <b>Location:</b> {lat:.4f}, {lon:.4f}<br>
            <hr>
            <b>OSM Features:</b><br>
            """
            
            for feature, count in sorted(osm_features.items())[:10]:  # Top 10 features
                popup_content += f"• {feature.title()}: {count}<br>"
            
            if show_predictions:
                # Get predictions
                existing_stations = list(self.trained_predictor.station_coords.keys())
                outflows = self.new_station_predictor.predict_flows_from_new_station(
                    station_id, existing_stations, hour, day_of_week
                )
                inflows = self.new_station_predictor.predict_flows_to_new_station(
                    existing_stations, station_id, hour, day_of_week
                )
                
                total_outflow = sum(outflows.values())
                total_inflow = sum(inflows.values())
                
                popup_content += f"""
                <hr>
                <b>Predicted Flows ({hour}:00, Day {day_of_week}):</b><br>
                • Total Outflow: {total_outflow:.1f}<br>
                • Total Inflow: {total_inflow:.1f}<br>
                • Net Flow: {total_outflow - total_inflow:.1f}
                """
                
                # Add flow lines
                self._add_prediction_flows(new_station_group, station_id, lat, lon, outflows, inflows)
            
            # Add new station marker
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"New {name}",
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(new_station_group)
            
            # Add OSM feature circle if requested
            if show_osm_features and osm_features:
                total_features = sum(osm_features.values())
                circle_radius = min(200, max(50, total_features * 5))
                
                folium.Circle(
                    [lat, lon],
                    radius=circle_radius,
                    popup=f"OSM Feature Density: {total_features}",
                    color='orange',
                    fill=True,
                    opacity=0.3
                ).add_to(new_station_group)
        
        # Add group to map
        new_station_group.add_to(m)
    
    def _add_prediction_flows(self, group: folium.FeatureGroup, station_id: str, 
                            lat: float, lon: float, outflows: Dict[str, float], 
                            inflows: Dict[str, float]):
        """Add flow prediction lines to map"""
        
        # Add top outflow connections
        top_outflows = sorted(outflows.items(), key=lambda x: x[1], reverse=True)[:3]
        for target_station, flow in top_outflows:
            if flow > 0 and target_station in self.trained_predictor.station_coords:
                target_lat, target_lon = self.trained_predictor.station_coords[target_station]
                
                line_weight = max(2, min(8, flow / 3))
                folium.PolyLine(
                    [[lat, lon], [target_lat, target_lon]],
                    weight=line_weight,
                    color='red',
                    opacity=0.6,
                    popup=f"Predicted outflow: {flow:.1f}"
                ).add_to(group)
        
        # Add top inflow connections  
        top_inflows = sorted(inflows.items(), key=lambda x: x[1], reverse=True)[:3]
        for source_station, flow in top_inflows:
            if flow > 0 and source_station in self.trained_predictor.station_coords:
                source_lat, source_lon = self.trained_predictor.station_coords[source_station]
                
                line_weight = max(2, min(8, flow / 3))
                folium.PolyLine(
                    [[source_lat, source_lon], [lat, lon]],
                    weight=line_weight,
                    color='green', 
                    opacity=0.6,
                    popup=f"Predicted inflow: {flow:.1f}"
                ).add_to(group)
    
    def _add_osm_feature_overlays(self, m: folium.Map):
        """Add OSM feature density overlays to map"""
        
        # Create heatmap data for different OSM features
        restaurant_data = []
        transport_data = []
        commercial_data = []
        
        # Collect data from existing stations
        for station_id, features in self.trained_predictor.station_features.items():
            if station_id not in self.trained_predictor.station_coords:
                continue
                
            lat, lon = self.trained_predictor.station_coords[station_id]
            
            # Extract feature values
            restaurants = features.get('cached_restaurants', features.get('osm_restaurants', 0))
            transport = features.get('cached_public_transport', features.get('osm_public_transport', 0))
            commercial = features.get('cached_shops', features.get('osm_shops', 0))
            
            if restaurants > 0:
                restaurant_data.append([lat, lon, restaurants])
            if transport > 0:
                transport_data.append([lat, lon, transport])
            if commercial > 0:
                commercial_data.append([lat, lon, commercial])
        
        # Add data from new stations
        for station_id, station in self.new_station_predictor.new_stations.items():
            lat, lon = station['lat'], station['lon']
            features = self.new_station_predictor.new_station_features.get(station_id, {})
            
            restaurants = features.get('cached_restaurants', features.get('osm_restaurants', 0))
            transport = features.get('cached_public_transport', features.get('osm_public_transport', 0))
            commercial = features.get('cached_shops', features.get('osm_shops', 0))
            
            if restaurants > 0:
                restaurant_data.append([lat, lon, restaurants])
            if transport > 0:
                transport_data.append([lat, lon, transport])
            if commercial > 0:
                commercial_data.append([lat, lon, commercial])
        
        # Add heatmap layers
        if restaurant_data:
            HeatMap(
                restaurant_data,
                name="Restaurant Density",
                radius=15,
                blur=15,
                max_zoom=18,
                overlay=True,
                control=True
            ).add_to(m)
        
        if transport_data:
            HeatMap(
                transport_data,
                name="Transport Density",
                radius=15,
                blur=15,
                max_zoom=18,
                overlay=True,
                control=True
            ).add_to(m)
        
        if commercial_data:
            HeatMap(
                commercial_data,
                name="Commercial Density",
                radius=15,
                blur=15,
                max_zoom=18,
                overlay=True,
                control=True
            ).add_to(m)
    
    def create_accuracy_plots(self, output_dir: str = "accuracy_plots"):
        """Create accuracy visualization plots"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.test_results:
            logger.warning("No test results available for plotting")
            return
        
        # Extract metrics
        metrics_data = []
        for station_id, result in self.test_results.items():
            if 'error' not in result:
                metrics_data.append({
                    'station_id': station_id,
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'r2_score': result['r2_score'],
                    'correlation': result['correlation'],
                    'n_comparisons': result['n_comparisons']
                })
        
        if not metrics_data:
            logger.warning("No valid metrics for plotting")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² score distribution
        axes[0, 0].hist(df['r2_score'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('R² Score Distribution')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_ylabel('Number of Stations')
        axes[0, 0].axvline(df['r2_score'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["r2_score"].mean():.3f}')
        axes[0, 0].legend()
        
        # RMSE vs MAE
        axes[0, 1].scatter(df['rmse'], df['mae'], alpha=0.7)
        axes[0, 1].set_title('RMSE vs MAE')
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_ylabel('MAE')
        
        # Accuracy by station
        df_sorted = df.sort_values('r2_score', ascending=True)
        axes[1, 0].barh(range(len(df_sorted)), df_sorted['r2_score'])
        axes[1, 0].set_yticks(range(len(df_sorted)))
        axes[1, 0].set_yticklabels(df_sorted['station_id'])
        axes[1, 0].set_title('R² Score by Station')
        axes[1, 0].set_xlabel('R² Score')
        
        # Correlation distribution
        axes[1, 1].hist(df['correlation'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_title('Correlation Distribution')
        axes[1, 1].set_xlabel('Correlation')
        axes[1, 1].set_ylabel('Number of Stations')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, "accuracy_summary.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved accuracy plots to {plot_file}")
    
    def export_results(self, output_dir: str = "enhanced_results"):
        """Export all results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export test results
        if self.test_results:
            test_df = pd.DataFrame([
                {
                    'station_id': sid,
                    'rmse': result.get('rmse', np.nan),
                    'mae': result.get('mae', np.nan),
                    'r2_score': result.get('r2_score', np.nan),
                    'correlation': result.get('correlation', np.nan),
                    'n_comparisons': result.get('n_comparisons', 0)
                }
                for sid, result in self.test_results.items()
                if 'error' not in result
            ])
            
            test_df.to_csv(os.path.join(output_dir, "test_results.csv"), index=False)
        
        # Export new station predictions
        for station_id in self.new_station_predictor.new_stations.keys():
            pred_df = self.new_station_predictor.export_predictions(
                station_id, 
                os.path.join(output_dir, f"predictions_{station_id}.csv")
            )
        
        logger.info(f"Exported results to {output_dir}/")

def demo_system():
    """Demo the enhanced system"""
    
    print("Enhanced Map-Integrated New Station Prediction Demo")
    print("=" * 55)
    
    print("\nSystem Features:")
    print("- New station prediction with OSM feature extraction")
    print("- Integration with existing optimized map system")  
    print("- Accuracy evaluation using existing stations as test cases")
    print("- Enhanced map visualizations with prediction flows")
    print("- OSM feature density heatmaps")
    print("- Comprehensive reporting and exports")
    
    print("\nUsage Instructions:")
    print("1. Train your GNN model: predictor = GNNFlowPredictor(trips_df)")
    print("2. Initialize enhanced system: enhanced = EnhancedMapIntegratedPredictor(predictor)")
    print("3. Add new stations: enhanced.add_new_stations_from_coordinates(coords)")
    print("4. Evaluate accuracy: enhanced.evaluate_new_station_accuracy(station_id)")
    print("5. Create enhanced map: enhanced.create_enhanced_osm_map(new_station_ids)")
    print("6. Export results: enhanced.export_results()")

if __name__ == "__main__":
    demo_system()
