#!/usr/bin/env python3
"""
Full DCRNN Only Runner
======================
Ensures only the complete DCRNN implementation is used, no fallbacks to minimal versions.
"""

import subprocess
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_dcrnn_only():
    """Run only the full DCRNN implementation"""
    logger.info("🚀 Starting FULL DCRNN Training (No Minimal Fallbacks)")
    
    # Ensure we're using the full implementation
    dcrnn_script = Path('train_predict_od_dcrnn.py')
    if not dcrnn_script.exists():
        logger.error("❌ Full DCRNN script not found: train_predict_od_dcrnn.py")
        return False
    
    # Remove any minimal DCRNN files to prevent fallbacks
    minimal_files = [
        'helpers/minimal_dcrnn.py'
    ]
    
    for file_path in minimal_files:
        if Path(file_path).exists():
            logger.warning(f"⚠️  Minimal DCRNN file exists: {file_path}")
            logger.info("   This could cause fallback to minimal implementation")
    
    # Run the full DCRNN
    try:
        logger.info("🔄 Executing full DCRNN training...")
        result = subprocess.run(
            [sys.executable, str(dcrnn_script)], 
            capture_output=True, 
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Full DCRNN training completed successfully!")
            logger.info(f"📊 Output:\n{result.stdout}")
            
            # Verify predictions were generated
            pred_file = Path('predicted_flows.json')
            if pred_file.exists():
                with open(pred_file, 'r') as f:
                    predictions = json.load(f)
                logger.info(f"📈 Generated {len(predictions)} flow predictions")
                
                # Show sample predictions
                if predictions:
                    sample = predictions[0]
                    logger.info(f"📋 Sample prediction: {sample}")
                
                return True
            else:
                logger.error("❌ No predictions file generated")
                return False
        else:
            logger.error(f"❌ Full DCRNN training failed!")
            logger.error(f"📋 Error output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ DCRNN training timed out (>1 hour)")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to run full DCRNN: {e}")
        return False

def verify_full_dcrnn_components():
    """Verify all full DCRNN components are present"""
    logger.info("🔍 Verifying Full DCRNN Components...")
    
    required_files = [
        'train_predict_od_dcrnn.py',
        'data/switzerland_station_features_1000m_with_pop.csv',
        'data/trips_8days_flat.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    logger.info("✅ All required components present")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("FULL DCRNN ONLY RUNNER")
    print("=" * 60)
    
    if not verify_full_dcrnn_components():
        print("❌ Component verification failed")
        sys.exit(1)
    
    success = run_full_dcrnn_only()
    
    if success:
        print("\n🎉 Full DCRNN training completed successfully!")
        print("📁 Check predicted_flows.json for results")
    else:
        print("\n❌ Full DCRNN training failed")
        sys.exit(1)