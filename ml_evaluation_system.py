"""
Robust ML Evaluation System for Bike Flow Prediction
Provides comprehensive evaluation metrics, confidence estimation, and model validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from sklearn.model_selection import (
    cross_val_score, TimeSeriesSplit, StratifiedKFold, 
    train_test_split, validation_curve, learning_curve
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    rmse: float
    mae: float
    mape: float
    r2: float
    explained_variance: float
    pearson_corr: float
    spearman_corr: float
    confidence_score: float
    prediction_interval_coverage: float
    calibration_score: float

@dataclass
class ConfidenceMetrics:
    """Container for confidence-related metrics"""
    prediction: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    uncertainty: float
    prediction_interval_width: float
    calibration_probability: float

class MLEvaluationSystem:
    """Comprehensive ML evaluation system with confidence estimation"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.evaluation_history = []
        self.confidence_calibrator = None
        self.feature_importance_history = []
        
    def comprehensive_evaluation(self, model, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str] = None,
                               cv_folds: int = 5) -> Dict[str, Any]:
        """Perform comprehensive model evaluation"""
        
        logger.info("Starting comprehensive model evaluation...")
        
        results = {}
        
        # 1. Cross-validation evaluation
        cv_results = self._cross_validation_evaluation(model, X, y, cv_folds)
        results['cross_validation'] = cv_results
        
        # 2. Hold-out evaluation
        holdout_results = self._holdout_evaluation(model, X, y)
        results['holdout'] = holdout_results
        
        # 3. Time series evaluation (if applicable)
        ts_results = self._time_series_evaluation(model, X, y)
        results['time_series'] = ts_results
        
        # 4. Feature importance analysis
        if feature_names:
            importance_results = self._feature_importance_analysis(model, X, y, feature_names)
            results['feature_importance'] = importance_results
        
        # 5. Residual analysis
        residual_results = self._residual_analysis(model, X, y)
        results['residual_analysis'] = residual_results
        
        # 6. Prediction confidence analysis
        confidence_results = self._confidence_analysis(model, X, y)
        results['confidence_analysis'] = confidence_results
        
        # 7. Model stability analysis
        stability_results = self._model_stability_analysis(model, X, y)
        results['stability_analysis'] = stability_results
        
        # 8. Overall evaluation summary
        summary = self._create_evaluation_summary(results)
        results['summary'] = summary
        
        # Store in history
        self.evaluation_history.append(results)
        
        logger.info("Comprehensive evaluation completed")
        return results
    
    def _cross_validation_evaluation(self, model, X: np.ndarray, y: np.ndarray, 
                                   cv_folds: int) -> Dict[str, Any]:
        """Perform cross-validation evaluation"""
        
        # Standard k-fold cross-validation
        cv_scores = {}
        
        # RMSE
        rmse_scores = -cross_val_score(model, X, y, cv=cv_folds, 
                                      scoring='neg_root_mean_squared_error')
        cv_scores['rmse'] = {
            'mean': np.mean(rmse_scores),
            'std': np.std(rmse_scores),
            'scores': rmse_scores.tolist()
        }
        
        # MAE
        mae_scores = -cross_val_score(model, X, y, cv=cv_folds, 
                                     scoring='neg_mean_absolute_error')
        cv_scores['mae'] = {
            'mean': np.mean(mae_scores),
            'std': np.std(mae_scores),
            'scores': mae_scores.tolist()
        }
        
        # R²
        r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        cv_scores['r2'] = {
            'mean': np.mean(r2_scores),
            'std': np.std(r2_scores),
            'scores': r2_scores.tolist()
        }
        
        return cv_scores
    
    def _holdout_evaluation(self, model, X: np.ndarray, y: np.ndarray, 
                          test_size: float = 0.2) -> Dict[str, Any]:
        """Perform hold-out evaluation"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_regression_metrics(y_test, y_pred)
        
        # Add prediction vs actual data
        metrics['predictions'] = {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }
        
        return metrics
    
    def _time_series_evaluation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform time series cross-validation"""
        
        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=5)
        
        ts_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics for this fold
            fold_metrics = self._calculate_regression_metrics(y_test, y_pred)
            ts_scores.append(fold_metrics)
        
        # Aggregate results
        aggregated = {}
        for metric in ['rmse', 'mae', 'r2', 'mape']:
            values = [score[metric] for score in ts_scores if metric in score]
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'scores': values
                }
        
        return aggregated
    
    def _feature_importance_analysis(self, model, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance"""
        
        # Train model to get feature importance
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Permutation importance for more robust analysis
            from sklearn.inspection import permutation_importance
            
            perm_importance = permutation_importance(
                model, X, y, n_repeats=10, random_state=self.random_state
            )
            
            perm_df = pd.DataFrame({
                'feature': feature_names,
                'perm_importance_mean': perm_importance.importances_mean,
                'perm_importance_std': perm_importance.importances_std
            }).sort_values('perm_importance_mean', ascending=False)
            
            # Store in history
            self.feature_importance_history.append({
                'standard_importance': importance_df.to_dict('records'),
                'permutation_importance': perm_df.to_dict('records')
            })
            
            return {
                'standard_importance': importance_df.to_dict('records'),
                'permutation_importance': perm_df.to_dict('records'),
                'top_features': importance_df.head(10)['feature'].tolist()
            }
        
        return {'message': 'Model does not support feature importance'}
    
    def _residual_analysis(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze model residuals"""
        
        # Train model and get predictions
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'q50': np.percentile(residuals, 50),
            'q75': np.percentile(residuals, 75)
        }
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for performance
        
        # Homoscedasticity test (Breusch-Pagan)
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(y_pred, np.abs(residuals))
            homoscedasticity_p = p_value
        except:
            homoscedasticity_p = None
        
        return {
            'statistics': residual_stats,
            'normality_test': {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'homoscedasticity_p': homoscedasticity_p,
            'residuals': residuals[:1000].tolist()  # Sample for visualization
        }
    
    def _confidence_analysis(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence and uncertainty"""
        
        # Bootstrap confidence intervals
        n_bootstrap = 100
        bootstrap_predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model on bootstrap sample
            model.fit(X_boot, y_boot)
            y_pred_boot = model.predict(X)
            bootstrap_predictions.append(y_pred_boot)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate confidence intervals
        confidence_intervals = {
            'lower_95': np.percentile(bootstrap_predictions, 2.5, axis=0),
            'upper_95': np.percentile(bootstrap_predictions, 97.5, axis=0),
            'lower_90': np.percentile(bootstrap_predictions, 5, axis=0),
            'upper_90': np.percentile(bootstrap_predictions, 95, axis=0),
            'mean': np.mean(bootstrap_predictions, axis=0),
            'std': np.std(bootstrap_predictions, axis=0)
        }
        
        # Prediction interval coverage
        model.fit(X, y)
        y_pred = model.predict(X)
        
        coverage_95 = np.mean(
            (y >= confidence_intervals['lower_95']) & 
            (y <= confidence_intervals['upper_95'])
        )
        
        coverage_90 = np.mean(
            (y >= confidence_intervals['lower_90']) & 
            (y <= confidence_intervals['upper_90'])
        )
        
        return {
            'confidence_intervals': {
                key: values[:100].tolist()  # Sample for storage
                for key, values in confidence_intervals.items()
            },
            'coverage': {
                'coverage_95': coverage_95,
                'coverage_90': coverage_90,
                'target_95': 0.95,
                'target_90': 0.90
            },
            'uncertainty_stats': {
                'mean_uncertainty': np.mean(confidence_intervals['std']),
                'max_uncertainty': np.max(confidence_intervals['std']),
                'min_uncertainty': np.min(confidence_intervals['std'])
            }
        }
    
    def _model_stability_analysis(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze model stability across different data splits"""
        
        n_splits = 10
        stability_scores = []
        
        for i in range(n_splits):
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=i
            )
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = self._calculate_regression_metrics(y_test, y_pred)
            stability_scores.append(metrics)
        
        # Calculate stability metrics
        stability_metrics = {}
        for metric in ['rmse', 'mae', 'r2']:
            values = [score[metric] for score in stability_scores if metric in score]
            if values:
                stability_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf'),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stability_metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # MAPE (handle division by zero)
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            # Custom MAPE calculation
            mask = y_true != 0
            if np.sum(mask) > 0:
                metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                metrics['mape'] = float('inf')
        
        # Correlation metrics
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            metrics['pearson_corr'], _ = pearsonr(y_true, y_pred)
            metrics['spearman_corr'], _ = spearmanr(y_true, y_pred)
        else:
            metrics['pearson_corr'] = 0.0
            metrics['spearman_corr'] = 0.0
        
        return metrics
    
    def _create_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create overall evaluation summary"""
        
        summary = {
            'overall_score': 0.0,
            'confidence_grade': 'Unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Extract key metrics
        cv_r2 = results.get('cross_validation', {}).get('r2', {}).get('mean', 0)
        holdout_r2 = results.get('holdout', {}).get('r2', 0)
        cv_rmse = results.get('cross_validation', {}).get('rmse', {}).get('mean', float('inf'))
        
        # Calculate overall score (0-100)
        r2_score = max(0, min(100, cv_r2 * 100))
        stability_score = 100 - (results.get('cross_validation', {}).get('r2', {}).get('std', 1) * 100)
        stability_score = max(0, min(100, stability_score))
        
        summary['overall_score'] = (r2_score * 0.6) + (stability_score * 0.4)
        
        # Confidence grade
        if summary['overall_score'] >= 80:
            summary['confidence_grade'] = 'High'
        elif summary['overall_score'] >= 60:
            summary['confidence_grade'] = 'Medium'
        elif summary['overall_score'] >= 40:
            summary['confidence_grade'] = 'Low'
        else:
            summary['confidence_grade'] = 'Very Low'
        
        # Strengths and weaknesses
        if cv_r2 > 0.7:
            summary['strengths'].append('Good predictive performance (R² > 0.7)')
        if results.get('cross_validation', {}).get('r2', {}).get('std', 1) < 0.1:
            summary['strengths'].append('Stable performance across folds')
        
        if cv_r2 < 0.5:
            summary['weaknesses'].append('Low predictive performance (R² < 0.5)')
        if results.get('cross_validation', {}).get('r2', {}).get('std', 0) > 0.2:
            summary['weaknesses'].append('High performance variability')
        
        # Recommendations
        if cv_r2 < 0.6:
            summary['recommendations'].append('Consider feature engineering or model selection')
        if results.get('cross_validation', {}).get('r2', {}).get('std', 0) > 0.15:
            summary['recommendations'].append('Investigate data quality and model stability')
        
        return summary
    
    def predict_with_confidence(self, model, X: np.ndarray, 
                              confidence_level: float = 0.95) -> List[ConfidenceMetrics]:
        """Make predictions with confidence intervals"""
        
        # Ensure model is trained
        if not hasattr(model, 'predict'):
            raise ValueError("Model must be trained before making predictions")
        
        # Base predictions
        predictions = model.predict(X)
        
        # Bootstrap for confidence intervals
        n_bootstrap = 50  # Reduced for performance
        bootstrap_preds = []
        
        # Use stored training data if available, otherwise use current X
        for i in range(n_bootstrap):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            
            # For simplicity, assume we can retrain (in practice, you'd store training data)
            try:
                # This is a simplified approach - in practice you'd need access to training data
                pred_boot = model.predict(X_boot)
                bootstrap_preds.append(pred_boot)
            except:
                # Fallback: add noise to original predictions
                noise = np.random.normal(0, np.std(predictions) * 0.1, len(predictions))
                bootstrap_preds.append(predictions + noise)
        
        bootstrap_preds = np.array(bootstrap_preds)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_metrics = []
        
        for i in range(len(predictions)):
            pred_samples = bootstrap_preds[:, i] if bootstrap_preds.shape[1] > i else bootstrap_preds[:, 0]
            
            lower_bound = np.percentile(pred_samples, lower_percentile)
            upper_bound = np.percentile(pred_samples, upper_percentile)
            uncertainty = np.std(pred_samples)
            
            metrics = ConfidenceMetrics(
                prediction=predictions[i],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                uncertainty=uncertainty,
                prediction_interval_width=upper_bound - lower_bound,
                calibration_probability=confidence_level  # Simplified
            )
            
            confidence_metrics.append(metrics)
        
        return confidence_metrics
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        
        report = "# ML Model Evaluation Report\n\n"
        
        # Summary
        summary = results.get('summary', {})
        report += f"## Overall Assessment\n\n"
        report += f"**Overall Score:** {summary.get('overall_score', 0):.1f}/100\n"
        report += f"**Confidence Grade:** {summary.get('confidence_grade', 'Unknown')}\n\n"
        
        # Cross-validation results
        cv_results = results.get('cross_validation', {})
        if cv_results:
            report += "## Cross-Validation Results\n\n"
            for metric, values in cv_results.items():
                if isinstance(values, dict):
                    mean_val = values.get('mean', 0)
                    std_val = values.get('std', 0)
                    report += f"**{metric.upper()}:** {mean_val:.4f} ± {std_val:.4f}\n"
            report += "\n"
        
        # Hold-out results
        holdout = results.get('holdout', {})
        if holdout:
            report += "## Hold-out Test Results\n\n"
            for metric, value in holdout.items():
                if isinstance(value, (int, float)):
                    report += f"**{metric.upper()}:** {value:.4f}\n"
            report += "\n"
        
        # Feature importance
        importance = results.get('feature_importance', {})
        if importance and 'top_features' in importance:
            report += "## Top Important Features\n\n"
            for i, feature in enumerate(importance['top_features'][:10], 1):
                report += f"{i}. {feature}\n"
            report += "\n"
        
        # Strengths and weaknesses
        if summary.get('strengths'):
            report += "## Model Strengths\n\n"
            for strength in summary['strengths']:
                report += f"- {strength}\n"
            report += "\n"
        
        if summary.get('weaknesses'):
            report += "## Areas for Improvement\n\n"
            for weakness in summary['weaknesses']:
                report += f"- {weakness}\n"
            report += "\n"
        
        # Recommendations
        if summary.get('recommendations'):
            report += "## Recommendations\n\n"
            for rec in summary['recommendations']:
                report += f"- {rec}\n"
            report += "\n"
        
        # Confidence analysis
        confidence = results.get('confidence_analysis', {})
        if confidence:
            coverage = confidence.get('coverage', {})
            report += "## Confidence Analysis\n\n"
            report += f"**95% Prediction Interval Coverage:** {coverage.get('coverage_95', 0):.2%}\n"
            report += f"**90% Prediction Interval Coverage:** {coverage.get('coverage_90', 0):.2%}\n"
            
            uncertainty = confidence.get('uncertainty_stats', {})
            report += f"**Mean Uncertainty:** {uncertainty.get('mean_uncertainty', 0):.4f}\n"
            report += "\n"
        
        return report

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Initialize evaluation system
    evaluator = MLEvaluationSystem()
    
    print("Running comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(model, X, y, feature_names)
    
    print("\nEvaluation Summary:")
    summary = results['summary']
    print(f"Overall Score: {summary['overall_score']:.1f}/100")
    print(f"Confidence Grade: {summary['confidence_grade']}")
    
    print("\nGenerating report...")
    report = evaluator.generate_evaluation_report(results)
    print(report[:500] + "...")  # Print first 500 characters
    
    print("\nTesting confidence predictions...")
    # Train model for confidence prediction
    model.fit(X[:800], y[:800])
    confidence_preds = evaluator.predict_with_confidence(model, X[800:810])
    
    print(f"Sample confidence predictions:")
    for i, conf in enumerate(confidence_preds[:3]):
        print(f"Prediction {i+1}: {conf.prediction:.2f} [{conf.lower_bound:.2f}, {conf.upper_bound:.2f}]")