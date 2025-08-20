#!/usr/bin/env python3
"""
GNN Baseline Testing with OSM Features
=====================================

Test GCN and GraphSAGE models using the extracted OSM features.
"""

import pandas as pd
import numpy as np
import torch
import logging
import sys
import os
import argparse
from typing import Dict, List

# Import GNN components with robust import logic
try:
    from helpers.gnn_flow_predictor import GNNFlowPredictor, GNNConfig, BikeFlowGNN
    from helpers.gnn_testing_framework import GNNTester
except ImportError:
    from gnn_flow_predictor import GNNFlowPredictor, GNNConfig, BikeFlowGNN
    from gnn_testing_framework import GNNTester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GNNBaselineRunner:
    def run_tabnet_baseline(self, radius):
        """Run TabNet regression using OSM features and aggregated flows."""
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.feature_selection import VarianceThreshold
        except ImportError:
            logger.error("TabNet not installed. Please install with: pip install pytorch-tabnet")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return None
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        # Feature selection: remove constant/low-variance features
        selector = VarianceThreshold(threshold=1e-5)
        X = selector.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y.values, test_size=0.2, random_state=42)
        # TabNet expects 2D targets for regression
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        model = TabNetRegressor(verbose=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=20)
        y_pred = model.predict(X_val).squeeze()
        import numpy as np
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"TabNet (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius, 'X_val': X_val, 'y_val': y_val.squeeze()}

    def run_stacking_ensemble(self, radius, base_models=None):
        """Stacking ensemble using predictions from base models (XGBoost, LightGBM, CatBoost, RF, MLP, TabNet)."""
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return None
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        # Prepare base models
        estimators = []
        if base_models is None:
            try:
                from xgboost import XGBRegressor
                estimators.append(('xgb', XGBRegressor(n_jobs=-1, random_state=42)))
            except ImportError:
                pass
            try:
                import lightgbm as lgb
                estimators.append(('lgb', lgb.LGBMRegressor(n_jobs=-1, random_state=42)))
            except ImportError:
                pass
            try:
                from catboost import CatBoostRegressor
                estimators.append(('cat', CatBoostRegressor(verbose=0, random_state=42)))
            except ImportError:
                pass
            try:
                from sklearn.ensemble import RandomForestRegressor
                estimators.append(('rf', RandomForestRegressor(n_jobs=-1, random_state=42)))
            except ImportError:
                pass
            try:
                from sklearn.neural_network import MLPRegressor
                estimators.append(('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)))
            except ImportError:
                pass
            # TabNetRegressor is not compatible with scikit-learn's StackingRegressor, so we skip it here.
        else:
            estimators = base_models
        if not estimators:
            logger.error("No base models available for stacking.")
            return None
        stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
        stack.fit(X_train, y_train)
        y_pred = stack.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"Stacking Ensemble (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': stack, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius}

    def run_blending_ensemble(self, radius, weights=None):
        """Blending ensemble: weighted average of predictions from available base models."""
        # Use all available tree/ML baselines for blending
        base_funcs = [
            self.run_xgboost_baseline,
            self.run_lightgbm_baseline,
            self.run_catboost_baseline,
            self.run_rf_baseline,
            self.run_mlp_baseline,
            self.run_tabnet_baseline
        ]
        preds = []
        y_true = None
        for func in base_funcs:
            result = func(radius)
            if result and 'model' in result:
                # Use the same validation set for all models
                if hasattr(result['model'], 'predict') and 'X_val' in result:
                    pred = result['model'].predict(result['X_val'])
                    preds.append(pred)
                    if y_true is None:
                        y_true = result['y_val']
        if not preds or y_true is None:
            logger.error("No valid base model predictions for blending.")
            return None
        preds = np.array(preds)
        if weights is None:
            weights = np.ones(preds.shape[0]) / preds.shape[0]
        y_blend = np.average(preds, axis=0, weights=weights)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_true, y_blend))
        mae = mean_absolute_error(y_true, y_blend)
        r2 = r2_score(y_true, y_blend)
        logger.info(f"Blending Ensemble (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius, 'weights': weights}

    def run_stgcn_baseline(self, radius=500, epochs=5):
        """Minimal ST-GCN-like baseline for demonstration. Requires torch and numpy."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            logger.error("PyTorch not installed. Please install with: pip install torch")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None or radius not in self.osm_features:
            logger.error("Aggregated flow data or OSM features not available.")
            return None
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow']).values.astype(np.float32)
        y = flows['flow'].values.astype(np.float32)
        # Fake adjacency: fully connected for demo
        A = np.ones((X.shape[0], X.shape[0]), dtype=np.float32)
        class SimpleGCN(nn.Module):
            def __init__(self, in_feats, out_feats):
                super().__init__()
                self.fc = nn.Linear(in_feats, out_feats)
            def forward(self, x, adj):
                h = torch.matmul(adj, x)
                return self.fc(h)
        model = SimpleGCN(X.shape[1], 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y).unsqueeze(1)
        A_tensor = torch.tensor(A)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_tensor, A_tensor)
            loss = loss_fn(out, y_tensor)
            loss.backward()
            optimizer.step()
        y_pred = model(X_tensor, A_tensor).detach().numpy().squeeze()
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        logger.info(f"ST-GCN (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius}

    def run_temporal_fusion_transformer(self, radius=500, epochs=5):
        """Minimal TFT-like baseline for demonstration. Requires torch."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            logger.error("PyTorch not installed. Please install with: pip install torch")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None or radius not in self.osm_features:
            logger.error("Aggregated flow data or OSM features not available.")
            return None
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        import numpy as np
        if not isinstance(X, np.ndarray):
            X = X.values
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        y = y.values.astype(np.float32)
        class SimpleMLP(nn.Module):
            def __init__(self, in_feats):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_feats, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            def forward(self, x):
                return self.net(x)
        model = SimpleMLP(X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y).unsqueeze(1)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(X_tensor)
            loss = loss_fn(out, y_tensor)
            loss.backward()
            optimizer.step()
        y_pred = model(X_tensor).detach().numpy().squeeze()
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        logger.info(f"Temporal Fusion Transformer (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius}
    def run_lightgbm_baseline(self, radius):
        """Run LightGBM regression using OSM features and aggregated flows, with feature selection and tuned params."""
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.feature_selection import VarianceThreshold
        except ImportError:
            logger.error("LightGBM not installed. Please install with: pip install lightgbm")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return None
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        # Feature selection: remove constant/low-variance features
        selector = VarianceThreshold(threshold=1e-5)
        X = selector.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'min_child_samples': [20, 50],
        }
        model = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, force_col_wise=True)
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"LightGBM (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': best_model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius, 'X_val': X_val, 'y_val': y_val}

    def run_catboost_baseline(self, radius):
        """Run CatBoost regression using OSM features and aggregated flows."""
        try:
            from catboost import CatBoostRegressor
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except ImportError:
            logger.error("CatBoost not installed. Please install with: pip install catboost")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return None
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = CatBoostRegressor(verbose=0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"CatBoost (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius}

    def run_rf_baseline(self, radius):
        """Run Random Forest regression using OSM features and aggregated flows."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except ImportError:
            logger.error("scikit-learn not installed. Please install with: pip install scikit-learn")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return None
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"RandomForest (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius}

    def run_mlp_baseline(self, radius):
        """Run MLP regression using OSM features and aggregated flows."""
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except ImportError:
            logger.error("scikit-learn not installed. Please install with: pip install scikit-learn")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return None
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"MLP (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius}
    def prepare_advanced_flow_samples(self, radius, seq_len=4, scaler=None, fit_scaler=True):
        """Prepare advanced samples for (source, destination, hour, day) with OSM, population, historical flows, and temporal features. Returns X as [num_samples, seq_len, num_features_per_step]."""
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return [], None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return [], None
        osm_features_df = self.osm_features[radius].set_index('station_id')
        # Optionally add population features if available
        pop_features = None
        if hasattr(self, 'population_features') and self.population_features and radius in self.population_features:
            pop_features = self.population_features[radius].set_index('station_id')
        # Prepare time-aware flows
        df = self.trips_df.copy()
        df['start_time_dt'] = pd.to_datetime(df['start_time'], format='%Y%m%d_%H%M%S', errors='coerce')
        df = df.dropna(subset=['start_time_dt'])
        df['hour'] = df['start_time_dt'].dt.hour
        df['dayofweek'] = df['start_time_dt'].dt.dayofweek
        # Group by (source, dest, hour, day)
        flow_df = df.groupby(['start_station_id', 'end_station_id', 'hour', 'dayofweek']).size().reset_index(name='flow')
        # Build a time series for each (source, dest) pair
        samples = []
        X_seq = []
        y_seq = []
        for (s, t), group in flow_df.groupby(['start_station_id', 'end_station_id']):
            group = group.sort_values(['dayofweek', 'hour'])
            flow_series = group['flow'].values.astype(np.float32)
            time_tuples = list(zip(group['dayofweek'], group['hour']))
            s_osm = osm_features_df.loc[s].drop(['lat', 'lon', 'coords'], errors='ignore').values.astype(np.float32) if s in osm_features_df.index else np.zeros(osm_features_df.shape[1] - sum([c in ['lat','lon','coords'] for c in osm_features_df.columns]), dtype=np.float32)
            t_osm = osm_features_df.loc[t].drop(['lat', 'lon', 'coords'], errors='ignore').values.astype(np.float32) if t in osm_features_df.index else np.zeros(osm_features_df.shape[1] - sum([c in ['lat','lon','coords'] for c in osm_features_df.columns]), dtype=np.float32)
            s_pop = pop_features.loc[s].values.astype(np.float32) if pop_features is not None and s in pop_features.index else np.array([], dtype=np.float32)
            t_pop = pop_features.loc[t].values.astype(np.float32) if pop_features is not None and t in pop_features.index else np.array([], dtype=np.float32)
            # For each possible prediction point
            for i in range(len(flow_series) - seq_len):
                seq_feats = []
                for j in range(seq_len):
                    # Features for each time step: [flow, s_osm, t_osm, s_pop, t_pop, hour_sin, hour_cos, dow_sin, dow_cos]
                    hist_flow = flow_series[i+j]
                    hist_day, hist_hour = time_tuples[i+j]
                    hour_sin = np.sin(2 * np.pi * hist_hour / 24)
                    hour_cos = np.cos(2 * np.pi * hist_hour / 24)
                    dow_sin = np.sin(2 * np.pi * hist_day / 7)
                    dow_cos = np.cos(2 * np.pi * hist_day / 7)
                    feats = np.concatenate([
                        [hist_flow], s_osm, t_osm, s_pop, t_pop, [hour_sin, hour_cos, dow_sin, dow_cos]
                    ])
                    seq_feats.append(feats)
                X_seq.append(seq_feats)
                y_seq.append(flow_series[i+seq_len])
                samples.append({
                    'source_id': s,
                    'target_id': t,
                    'target': flow_series[i+seq_len],
                    'hour': time_tuples[i+seq_len][1],
                    'dayofweek': time_tuples[i+seq_len][0]
                })
        X_seq = np.array(X_seq, dtype=np.float32) if X_seq else np.empty((0, seq_len, 1))
        y_seq = np.array(y_seq, dtype=np.float32) if y_seq else np.empty((0,))
        # Normalize features (fit scaler on flattened X_seq)
        if fit_scaler and X_seq.shape[0] > 0:
            scaler = StandardScaler()
            scaler.fit(X_seq.reshape(-1, X_seq.shape[2]))
        if scaler is not None and X_seq.shape[0] > 0:
            X_seq = scaler.transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)
        return (X_seq, y_seq, samples, scaler)
    def run_xgboost_baseline(self, radius):
        """Run XGBoost regression using OSM features and aggregated flows."""
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except ImportError:
            logger.error("XGBoost not installed. Please install with: pip install xgboost")
            return None
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return None
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return None
        # Merge OSM features for start and end stations
        osm = self.osm_features[radius]
        flows = self.flow_df.copy()
        # Merge start features
        flows = flows.merge(osm.add_prefix('start_'), left_on='start_station_id', right_on='start_station_id')
        # Merge end features
        flows = flows.merge(osm.add_prefix('end_'), left_on='end_station_id', right_on='end_station_id')
        # Drop non-feature columns
        drop_cols = [c for c in flows.columns if 'coords' in c or c.endswith('_id')]
        X = flows.drop(columns=drop_cols + ['flow'])
        y = flows['flow']
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        # Hyperparameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
        }
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"XGBoost (radius={radius}m): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
        return {'model': best_model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'radius': radius}
    def prepare_basic_flow_samples(self, radius):
        """Prepare training samples from aggregated flows (no time granularity)."""
        if not hasattr(self, 'flow_df') or self.flow_df is None:
            logger.error("Aggregated flow data not available.")
            return []
        if radius not in self.osm_features:
            logger.error(f"No OSM features for radius {radius}")
            return []
        osm_features_df = self.osm_features[radius]
        station_ids = list(osm_features_df['station_id'])
        station_to_idx = {station_id: idx for idx, station_id in enumerate(station_ids)}
        samples = []
        for _, row in self.flow_df.iterrows():
            s = row['start_station_id']
            t = row['end_station_id']
            flow = row['flow']
            if s in station_to_idx and t in station_to_idx:
                samples.append({
                    'source_idx': station_to_idx[s],
                    'target_idx': station_to_idx[t],
                    'flow': flow,
                    'time_vec': [0.5, 0.5]  # dummy if not using time
                })
        logger.info(f"Prepared {len(samples)} basic flow samples for radius {radius}")
        return samples
    def aggregate_flows(self, by_time=False):
        """Aggregate trips to compute flows between station pairs. Optionally by hour/day."""
        if self.trips_df is None:
            logger.error("Trip data not loaded.")
            return None
        df = self.trips_df.copy()
        # Parse time columns if needed
        if by_time:
            df['start_time_dt'] = pd.to_datetime(df['start_time'], format='%Y%m%d_%H%M%S', errors='coerce')
            df['hour'] = df['start_time_dt'].dt.hour
            df['dayofweek'] = df['start_time_dt'].dt.dayofweek
            group_cols = ['start_station_id', 'end_station_id', 'hour', 'dayofweek']
        else:
            group_cols = ['start_station_id', 'end_station_id']
        # Count trips (flows)
        flow_df = df.groupby(group_cols).size().reset_index(name='flow')
        logger.info(f"Aggregated flows: {len(flow_df)} rows (by_time={by_time})")
        return flow_df
    def check_data_quality(self, df=None, label="DATA QUALITY CHECK"):
        """Analyze data quality: missing values, variance, outliers, and feature-target correlation."""
        logger.info(f"\n===== {label} =====")
        if df is None:
            df = self.trips_df
        if df is None:
            logger.error("No data loaded.")
            return
        # Missing values
        missing = df.isnull().sum()
        logger.info(f"Missing values per column:\n{missing[missing > 0] if missing.sum() > 0 else 'None'}")
        # Low variance features
        numeric = df.select_dtypes(include=[np.number])
        low_var = numeric.var()[numeric.var() < 1e-5]
        logger.info(f"Low variance features:\n{low_var if not low_var.empty else 'None'}")
        # Outliers (z-score > 4)
        zscores = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0))
        outlier_counts = (zscores > 4).sum()
        logger.info(f"Outlier counts per feature (z-score > 4):\n{outlier_counts[outlier_counts > 0] if outlier_counts.sum() > 0 else 'None'}")
        # Feature-target correlation
        target_col = None
        for col in ['flow', 'target', 'label', 'y']:
            if col in numeric.columns:
                target_col = col
                break
        if target_col:
            corrs = numeric.corr()[target_col].drop(target_col)
            logger.info(f"Feature-target correlations (Pearson):\n{corrs}")
        else:
            logger.warning("No target column found for correlation analysis.")
        logger.info(f"===== END {label} =====\n")
    """Run GNN baselines with OSM features"""
    
    def __init__(self):
        self.trips_df = None
        self.osm_features = {}
        self.results = {}
        
    def load_data(self):
        """Load trip data and OSM features"""
        logger.info("Loading datasets for GNN testing...")
        
        # Load trips
        self.trips_df = pd.read_csv('data/trips_8days_flat.csv')
        logger.info(f"Loaded {len(self.trips_df)} trip records")
        
        # Load OSM features for different radii
        radii = [500, 1000, 1500]
        for radius in radii:
            try:
                osm_file = f'cache/ml_ready/switzerland_station_features_{radius}m.csv'
                osm_df = pd.read_csv(osm_file)
                self.osm_features[radius] = osm_df
                logger.info(f"Loaded OSM features for {radius}m radius: {len(osm_df)} stations")
            except FileNotFoundError:
                logger.warning(f"OSM features file not found for {radius}m radius")
        
        # Aggregate flows and store as attribute for use as target
        self.flow_df = self.aggregate_flows(by_time=False)
        return True
    
    def create_gnn_configs(self, device: torch.device = None) -> List[GNNConfig]:
        """Create different GNN configurations to test"""
        configs = []
        # Adjust batch size based on device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 64 if device.type == 'cuda' else 32
        epochs = 150 if device.type == 'cuda' else 100  # More epochs on GPU

        # DCRNN (stub, not implemented)
        dcrnn_config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.01,
            batch_size=batch_size,
            epochs=epochs,
            gnn_type="DCRNN",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('DCRNN_Baseline', dcrnn_config))
        
        # GCN Baseline - Simple and robust
        gcn_config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.01,
            batch_size=batch_size,
            epochs=epochs,
            gnn_type="GCN",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('GCN_Baseline', gcn_config))
        
        # GraphSAGE - Good for larger graphs
        sage_config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.01,
            batch_size=batch_size,
            epochs=epochs,
            gnn_type="GraphSAGE",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('GraphSAGE_Baseline', sage_config))
        
        # GAT - Attention-based (if you want to test)
        gat_config = GNNConfig(
            hidden_dim=32,  # Smaller due to multi-head attention
            num_layers=2,
            dropout=0.3,
            learning_rate=0.01,
            batch_size=batch_size // 2,  # Smaller batch for GAT due to memory usage
            epochs=epochs,
            attention_heads=4,
            gnn_type="GAT",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('GAT_Baseline', gat_config))
        
        # GIN - Graph Isomorphism Network
        gin_config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.01,
            batch_size=batch_size,
            epochs=epochs,
            gnn_type="GIN",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('GIN_Baseline', gin_config))
        
        # Graph Transformer
        transformer_config = GNNConfig(
            hidden_dim=32,
            num_layers=2,
            dropout=0.3,
            learning_rate=0.01,
            batch_size=batch_size // 2,
            epochs=epochs,
            attention_heads=4,
            gnn_type="Transformer",
            edge_features=True,
            use_batch_norm=True,
            use_residual=False
        )
        configs.append(('Transformer_Baseline', transformer_config))
        
        return configs
    
    def run_gnn_baseline(self, config_name: str, config: GNNConfig, radius: int, device: torch.device = None) -> Dict:
        if config.gnn_type == "DCRNN":
            # Use advanced samples for DCRNN: (source, dest, time) with all features
            seq_len = getattr(config, 'seq_len', 4)
            X_seq, y_seq, samples, scaler = self.prepare_advanced_flow_samples(radius, seq_len=seq_len, scaler=None, fit_scaler=True)
            if X_seq.shape[0] == 0:
                logger.error("No DCRNN training samples generated (not enough data)")
                return None
            # Train/val/test split by time (simulate by splitting sequentially)
            num_samples = X_seq.shape[0]
            train_idx = int(num_samples * 0.7)
            val_idx = int(num_samples * 0.85)
            x_train, x_val, x_test = X_seq[:train_idx], X_seq[train_idx:val_idx], X_seq[val_idx:]
            y_train, y_val, y_test = y_seq[:train_idx], y_seq[train_idx:val_idx], y_seq[val_idx:]
            # Convert to torch tensors
            x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
            y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
            x_val = torch.tensor(x_val, dtype=torch.float32, device=device)
            y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
            x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
            y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
            # Setup model
            gnn_predictor = GNNFlowPredictor(self.trips_df, config, use_cached_features=True, cache_file=None)
            gnn_predictor.device = device
            node_features_dim = x_train.shape[2]
            edge_features_dim = 0
            dcrnn_model = BikeFlowGNN(config, node_features_dim, edge_features_dim)
            gnn_predictor.dcrnn = dcrnn_model.dcrnn
            gnn_predictor.model = dcrnn_model.dcrnn
            if gnn_predictor.model is None:
                logger.error("DCRNN model is not initialized.")
                return None
            gnn_predictor.model.to(device)
            # Training loop
            optimizer = torch.optim.Adam(gnn_predictor.model.parameters(), lr=0.01)
            loss_fn = torch.nn.MSELoss()
            gnn_predictor.model.train()
            for epoch in range(30):
                optimizer.zero_grad()
                pred = gnn_predictor.model(x_seq=x_train)
                loss = loss_fn(pred, y_train)
                loss.backward()
                optimizer.step()
            # Evaluation on test set
            gnn_predictor.model.eval()
            with torch.no_grad():
                pred = gnn_predictor.model(x_seq=x_test)
            pred_np = pred.cpu().numpy()
            y_np = y_test.cpu().numpy()
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_np, pred_np))
            mae = mean_absolute_error(y_np, pred_np)
            r2 = r2_score(y_np, pred_np)
            logger.info(f"DCRNN output shape: {pred.shape}")
            return {'config_name': config_name, 'radius': radius, 'gnn_type': config.gnn_type, 'training_samples': len(y_np), 'test_rmse': rmse, 'test_mae': mae, 'test_r2': r2, 'num_features': x_train.shape[2], 'config': config.__dict__}
        """Run a single GNN configuration"""
        logger.info(f"\\n{'='*60}")
        logger.info(f"TESTING {config_name} with {radius}m OSM features")
        logger.info(f"{'='*60}")
        
        try:
            # Set default device if not provided
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
            # Create GNN predictor and instruct it to load the explicit per-radius cache file
            cache_file = f'cache/ml_ready/switzerland_station_features_{radius}m.csv'
            gnn_predictor = GNNFlowPredictor(self.trips_df, config, use_cached_features=True, cache_file=cache_file)
            
            # Set device on the predictor
            gnn_predictor.device = device
            
            # Add OSM features if available
            if radius in self.osm_features:
                logger.info(f"Adding OSM features from {radius}m radius...")
                
                # Get OSM features and directly set them on the predictor
                osm_features_df = self.osm_features[radius]
                
                # Extract station coordinates from OSM features
                logger.info(f"Building station network from OSM features...")
                gnn_predictor.station_coords = {}
                gnn_predictor.station_features = {}
                
                for _, row in osm_features_df.iterrows():
                    station_id = row['station_id']
                    
                    # Extract coordinates
                    gnn_predictor.station_coords[station_id] = (row['lat'], row['lon'])
                    
                    # Extract all feature columns (excluding station_id, lat, lon, coords)
                    features = {}
                    for col in osm_features_df.columns:
                        if col not in ['station_id', 'lat', 'lon', 'coords']:
                            try:
                                # Only include numeric features
                                val = float(row[col])
                                features[col] = val
                            except (ValueError, TypeError):
                                # Skip non-numeric columns
                                continue
                    
                    gnn_predictor.station_features[station_id] = features
                
                logger.info(f"Loaded {len(gnn_predictor.station_coords)} stations with {len(features)} features each")
                
                # Build graph structure directly
                logger.info("Building graph structure...")
                station_to_idx, feature_names = gnn_predictor.build_graph_structure()
                
                # Prepare training data
                logger.info("Preparing training data...")
                # Use aggregated flows for basic baseline, else fallback to time-aware
                if hasattr(self, 'flow_df') and self.flow_df is not None:
                    training_samples = self.prepare_basic_flow_samples(radius)
                else:
                    training_samples = gnn_predictor.prepare_training_data()

                if len(training_samples) == 0:
                    logger.warning("No training samples generated")
                    return None
                
                # Train the model
                logger.info(f"Training GNN model with {len(training_samples)} samples...")
                gnn_predictor.train_gnn_model(training_samples)
                
                # Basic evaluation - predict some flows and calculate metrics
                logger.info("Evaluating model...")
                test_metrics = self.evaluate_gnn_model(gnn_predictor, training_samples[:100])
                
                # Combine results
                results = {
                    'config_name': config_name,
                    'radius': radius,
                    'gnn_type': config.gnn_type,
                    'training_samples': len(training_samples),
                    'test_rmse': test_metrics.get('rmse', 0),
                    'test_mae': test_metrics.get('mae', 0),
                    'test_r2': test_metrics.get('r2', 0),
                    'num_stations': len(station_to_idx),
                    'num_features': len(feature_names),
                    'config': config.__dict__
                }
                
                logger.info(f"Results: RMSE={results['test_rmse']:.2f}, MAE={results['test_mae']:.2f}, R²={results['test_r2']:.3f}")
                return results
                
            else:
                logger.error(f"No OSM features available for {radius}m radius")
                return None
                
        except Exception as e:
            logger.error(f"Error running {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_gnn_model(self, gnn_predictor, test_samples):
        """Evaluate GNN model on test samples"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        predictions = []
        actuals = []

        gnn_predictor.model.eval()

        graph_data = gnn_predictor.graph_data.to(gnn_predictor.device)

        # Create batch from test samples
        source_indices = torch.tensor([sample['source_idx'] for sample in test_samples], dtype=torch.long).to(gnn_predictor.device)
        target_indices = torch.tensor([sample['target_idx'] for sample in test_samples], dtype=torch.long).to(gnn_predictor.device)
        flows = torch.tensor([sample['flow'] for sample in test_samples], dtype=torch.float32)
        # Prepare time features if present
        time_feats = torch.tensor([sample.get('time_vec', [0.5, 0.5]) for sample in test_samples], dtype=torch.float32).to(gnn_predictor.device)

        # Get predictions
        pred_flows = gnn_predictor.model(
            graph_data.x,
            graph_data.edge_index,
            None,  # edge_attr
            source_indices,
            target_indices,
            time_feats
        )

        predictions = pred_flows.detach().cpu().numpy().flatten()
        actuals = flows.numpy().flatten()

        # Clear GPU memory
        if gnn_predictor.device.type == 'cuda':
            del graph_data, source_indices, target_indices, pred_flows, time_feats
            torch.cuda.empty_cache()

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def run_all_baselines(self, device: torch.device = None):
        """Run all GNN baselines with different OSM radii"""
        logger.info("🚀 Starting GNN Baseline Testing with OSM Features")
        
        # Set default device if not provided
        
        logger.info(f"Using device: {device}")
        
        if not self.load_data():
            logger.error("Failed to load data")
            return
        self.check_data_quality()
        
        configs = self.create_gnn_configs(device)
        radii = [500, 1000, 1500]  # Test all radii
        
        all_results = []
        
        # Test all configurations
        for config_name, config in configs:
            for radius in radii:
                if radius in self.osm_features:
                    result = self.run_gnn_baseline(config_name, config, radius, device)
                    if result:
                        all_results.append(result)
                        # Store in class results
                        key = f"{config_name}_{radius}m"
                        self.results[key] = result
                        # Clear GPU cache after each run to prevent memory issues
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()

        # Run XGBoost baselines for each radius
        for radius in radii:
            if radius in self.osm_features:
                xgb_result = self.run_xgboost_baseline(radius)
                if xgb_result:
                    all_results.append({
                        'config_name': 'XGBoost_Baseline',
                        'radius': radius,
                        'gnn_type': 'XGBoost',
                        'training_samples': None,
                        'test_rmse': xgb_result['rmse'],
                        'test_mae': xgb_result['mae'],
                        'test_r2': xgb_result['r2'],
                        'num_stations': None,
                        'num_features': None,
                        'config': None
                    })

        # Save results
        self.save_results(all_results)
        self.print_summary(all_results)
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Save results to CSV"""
        if not results:
            logger.warning("No results to save")
            return
            
        results_df = pd.DataFrame(results)
        results_file = f"results/gnn_baseline_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        os.makedirs('results', exist_ok=True)
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary of results"""
        if not results:
            return
            
        logger.info("\\n" + "="*80)
        logger.info("GNN BASELINE RESULTS SUMMARY")
        logger.info("="*80)
        
        df = pd.DataFrame(results)
        
        # Group by GNN type and radius
        for gnn_type in df['gnn_type'].unique():
            logger.info(f"\\n📊 {gnn_type} Results:")
            gnn_results = df[df['gnn_type'] == gnn_type]
            
            for _, row in gnn_results.iterrows():
                logger.info(f"  {row['radius']}m: RMSE={row['test_rmse']:.2f}, MAE={row['test_mae']:.2f}, R²={row['test_r2']:.3f}")
        
        # Best results
        best_result = df.loc[df['test_r2'].idxmax()]
        logger.info(f"\\nBEST RESULT:")
        logger.info(f"  Model: {best_result['config_name']} ({best_result['radius']}m)")
        logger.info(f"  RMSE: {best_result['test_rmse']:.2f}")
        logger.info(f"  MAE: {best_result['test_mae']:.2f}")
        logger.info(f"  R²: {best_result['test_r2']:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GNN and ML baselines for bike flow prediction.")
    parser.add_argument('--model', type=str, default=None, help='Name of the model to run (e.g., GCN_Baseline, GraphSAGE_Baseline, GAT_Baseline, GIN_Baseline, Transformer_Baseline, XGBoost_Baseline). If not set, runs all models.')
    return parser.parse_args()

def main():
    """Main execution"""
    args = parse_args()
    # Check if PyTorch Geometric is available
    try:
        import torch_geometric
        logger.info(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        logger.error("PyTorch Geometric not installed. Please install with: pip install torch-geometric")
        return
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    runner = GNNBaselineRunner()
    if args.model is None:
        # Interactive menu for model selection
        runner.load_data()
        runner.check_data_quality(label="RAW TRIPS DATA QUALITY CHECK")
        if hasattr(runner, 'flow_df') and runner.flow_df is not None:
            runner.check_data_quality(runner.flow_df, label="AGGREGATED FLOW DATA QUALITY CHECK")
        configs = runner.create_gnn_configs(device)
        # Add all available models to the menu
        model_names = [name.replace('_Baseline', '') for name, _ in configs] + [
            'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'MLP', 'TabNet',
            'Stacking', 'Blending', 'ST-GCN', 'TemporalFusionTransformer', 'All', 'Clear Screen & Return to Menu']
        while True:
            print("\nSelect a model to run:")
            for i, name in enumerate(model_names, 1):
                print(f"  {i}. {name}")
            selection = input("Enter the number of the model to run (or 'All' to run all): ").strip()
            if selection.lower() == 'all' or selection == str(model_names.index('All')+1):
                runner.run_all_baselines(device)
                logger.info("🎉 GNN baseline testing completed!")
                return
            if selection == str(model_names.index('Clear Screen & Return to Menu')+1):
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            try:
                idx = int(selection) - 1
                if idx < 0 or idx >= len(model_names):
                    raise ValueError
                selected_model = model_names[idx]
            except Exception:
                print("Invalid selection. Exiting.")
                return
            if selected_model == 'Clear Screen & Return to Menu':
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            # Set args.model for downstream logic
            args.model = selected_model
            break
    # Now run as if --model was provided
    runner.load_data()
    runner.check_data_quality(label="RAW TRIPS DATA QUALITY CHECK")
    if hasattr(runner, 'flow_df') and runner.flow_df is not None:
        runner.check_data_quality(runner.flow_df, label="AGGREGATED FLOW DATA QUALITY CHECK")
    configs = runner.create_gnn_configs(device)
    radii = [500, 1000, 1500]
    found = False
    model_arg = args.model.lower().replace('_baseline', '')
    # Dispatch logic for all models
    if model_arg == 'xgboost':
        for radius in radii:
            if radius in runner.osm_features:
                xgb_result = runner.run_xgboost_baseline(radius)
                if xgb_result:
                    print(f"XGBoost (radius={radius}): RMSE={xgb_result['rmse']:.2f}, MAE={xgb_result['mae']:.2f}, R2={xgb_result['r2']:.3f}")
                    found = True
    elif model_arg == 'lightgbm':
        for radius in radii:
            if radius in runner.osm_features:
                lgb_result = runner.run_lightgbm_baseline(radius)
                if lgb_result:
                    print(f"LightGBM (radius={radius}): RMSE={lgb_result['rmse']:.2f}, MAE={lgb_result['mae']:.2f}, R2={lgb_result['r2']:.3f}")
                    found = True
    elif model_arg == 'catboost':
        for radius in radii:
            if radius in runner.osm_features:
                cat_result = runner.run_catboost_baseline(radius)
                if cat_result:
                    print(f"CatBoost (radius={radius}): RMSE={cat_result['rmse']:.2f}, MAE={cat_result['mae']:.2f}, R2={cat_result['r2']:.3f}")
                    found = True
    elif model_arg == 'randomforest':
        for radius in radii:
            if radius in runner.osm_features:
                rf_result = runner.run_rf_baseline(radius)
                if rf_result:
                    print(f"RandomForest (radius={radius}): RMSE={rf_result['rmse']:.2f}, MAE={rf_result['mae']:.2f}, R2={rf_result['r2']:.3f}")
                    found = True
    elif model_arg == 'mlp':
        for radius in radii:
            if radius in runner.osm_features:
                mlp_result = runner.run_mlp_baseline(radius)
                if mlp_result:
                    print(f"MLP (radius={radius}): RMSE={mlp_result['rmse']:.2f}, MAE={mlp_result['mae']:.2f}, R2={mlp_result['r2']:.3f}")
                    found = True
    elif model_arg == 'tabnet':
        for radius in radii:
            if radius in runner.osm_features:
                tabnet_result = runner.run_tabnet_baseline(radius)
                if tabnet_result:
                    print(f"TabNet (radius={radius}): RMSE={tabnet_result['rmse']:.2f}, MAE={tabnet_result['mae']:.2f}, R2={tabnet_result['r2']:.3f}")
                    found = True
    elif model_arg == 'stacking':
        for radius in radii:
            if radius in runner.osm_features:
                stack_result = runner.run_stacking_ensemble(radius)
                if stack_result:
                    print(f"Stacking Ensemble (radius={radius}): RMSE={stack_result['rmse']:.2f}, MAE={stack_result['mae']:.2f}, R2={stack_result['r2']:.3f}")
                    found = True
    elif model_arg == 'blending':
        print("Blending requires predictions from multiple models. Please run base models and call blending manually.")
        found = True
    elif model_arg == 'st-gcn':
        runner.run_stgcn_baseline()
        found = True
    elif model_arg == 'temporalfusiontransformer':
        runner.run_temporal_fusion_transformer()
        found = True
    else:
        for config_name, config in configs:
            config_key = config_name.lower().replace('_baseline', '')
            if config_key == model_arg:
                for radius in radii:
                    if radius in runner.osm_features:
                        result = runner.run_gnn_baseline(config_name, config, radius, device)
                        if result:
                            rmse = f"{result['test_rmse']:.2f}" if result['test_rmse'] is not None else "N/A"
                            mae = f"{result['test_mae']:.2f}" if result['test_mae'] is not None else "N/A"
                            r2 = f"{result['test_r2']:.3f}" if result['test_r2'] is not None else "N/A"
                            print(f"{config_name} (radius={radius}): RMSE={rmse}, MAE={mae}, R2={r2}")
                            found = True
    if not found:
        print(f"Model '{args.model}' not found or no results produced.")

if __name__ == "__main__":
    main()
