"""
Full DCRNN Configuration
========================
Configuration that ensures only full DCRNN implementation is used
"""

import torch
from dataclasses import dataclass
from typing import Optional
import logging
import os


@dataclass
class FullDCRNNConfig:
    """Configuration for Full DCRNN only - no minimal fallbacks"""
    
    # Model Architecture
    hidden_dim: int = 64
    num_layers: int = 2
    seq_len: int = 6  # Historical timesteps
    predict_horizon: int = 1
    
    # Training
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 1
    
    # Data Processing
    max_stations: int = 714
    max_timesteps: int = 48
    chunk_size: int = 100  # Process stations in chunks
    
    # Graph Structure
    adjacency_threshold_percentile: int = 5  # Connect closest 5% of stations
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Files
    station_features_file: str = "data/switzerland_station_features_1000m_with_pop.csv"
    trips_file: str = "data/trips_8days_flat.csv"
    output_file: str = "predicted_flows.json"
    
    # Validation
    enforce_full_implementation: bool = True  # Fail if minimal version is used
    
    # In full_dcrnn_config.py, modify the validate method around line 55
    def validate(self):
        """Validate that we're using full DCRNN components only"""
        import sys
        
        # Check for minimal DCRNN files that shouldn't exist
        minimal_files = [
            'helpers/minimal_dcrnn.py',
            'minimal_dcrnn_model.py'
        ]
        
        for file_path in minimal_files:
            if os.path.exists(file_path):
                logging.warning(f"Minimal DCRNN file exists: {file_path}")
        
        # Check loaded modules - be more specific about what we're looking for
        problematic_modules = []
        for name, module in sys.modules.items():
            # Only check modules that are likely our custom modules
            if (name.startswith('minimal_dcrnn') or 
                name.startswith('helpers.minimal_dcrnn') or
                (hasattr(module, '__file__') and module.__file__ and 
                'minimal_dcrnn' in module.__file__.lower())):
                problematic_modules.append(name)
        
        if problematic_modules:
            raise RuntimeError(f"Minimal DCRNN modules detected: {problematic_modules}. Use full implementation only!")
        
        logging.info("✓ Full DCRNN validation passed")
        
        return True

def get_full_dcrnn_config() -> FullDCRNNConfig:
    """Get validated full DCRNN configuration"""
    config = FullDCRNNConfig()
    config.validate()
    return config