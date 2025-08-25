"""
Full DCRNN Configuration
========================
Configuration that ensures only full DCRNN implementation is used
"""

import torch
from dataclasses import dataclass
from typing import Optional

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
    
    def validate(self):
        """Validate configuration for full DCRNN"""
        if self.enforce_full_implementation:
            # Check that minimal DCRNN is not being used
            import inspect
            import sys
            
            # Look for minimal implementations in loaded modules
            for name, module in sys.modules.items():
                if hasattr(module, 'SimpleDCRNNCell') or hasattr(module, 'MinimalDCRNN'):
                    raise RuntimeError(f"❌ Minimal DCRNN detected in {name}. Use full implementation only!")
            
            print("✅ Full DCRNN configuration validated - no minimal implementations detected")
        
        return True

def get_full_dcrnn_config() -> FullDCRNNConfig:
    """Get validated full DCRNN configuration"""
    config = FullDCRNNConfig()
    config.validate()
    return config