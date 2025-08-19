#!/usr/bin/env python3
"""
Fix script for time features error in GNN code
"""

import os
import sys
import shutil
import re

def backup_file(filepath):
    """Create a backup of the file"""
    backup_path = f"{filepath}.bak"
    try:
        shutil.copy2(filepath, backup_path)
        print(f"Created backup at {backup_path}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def fix_run_gnn_baselines():
    """Fix the run_gnn_baselines.py file"""
    # Paths - adjust as needed
    base_path = "/home/ubuntu/code/EPFL_SURF_2025"
    filepath = os.path.join(base_path, "helpers/run_gnn_baselines.py")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    
    # Create backup
    if not backup_file(filepath):
        return False
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find the problematic code block
    pattern = r"(\s+)# Get predictions\s+pred_flows = gnn_predictor\.model\(\s+graph_data\.x,\s+graph_data\.edge_index,\s+graph_data\.edge_attr,\s+source_indices,\s+target_indices\s+\)"
    
    # Replacement with time features
    replacement = r"\1# Extract time features\n\1time_vecs = [sample.get(\"time_vec\", [0.5, 0.5]) for sample in test_samples]\n\1time_feats = torch.tensor(time_vecs, dtype=torch.float32).to(gnn_predictor.device)\n\n\1# Get predictions with time features\n\1pred_flows = gnn_predictor.model(\n\1    graph_data.x,\n\1    graph_data.edge_index,\n\1    graph_data.edge_attr,\n\1    source_indices,\n\1    target_indices,\n\1    time_feats\n\1)"
    
    # Apply the fix
    new_content = re.sub(pattern, replacement, content)
    
    # Check if any changes were made
    if new_content == content:
        print("No changes made - pattern not found. Using alternate approach.")
        
        # Try a more direct approach by looking for key lines
        lines = content.split('\n')
        found = False
        
        for i, line in enumerate(lines):
            if "pred_flows = gnn_predictor.model(" in line:
                # Count indentation
                indent = len(line) - len(line.lstrip())
                indent_str = " " * indent
                
                # Find the closing parenthesis
                j = i
                while j < len(lines) and ")" not in lines[j]:
                    j += 1
                
                if j < len(lines):
                    # Insert time features extraction before model call
                    time_extract = [
                        f"{indent_str}# Extract time features",
                        f"{indent_str}time_vecs = [sample.get(\"time_vec\", [0.5, 0.5]) for sample in test_samples]",
                        f"{indent_str}time_feats = torch.tensor(time_vecs, dtype=torch.float32).to(gnn_predictor.device)",
                        ""
                    ]
                    
                    # Add time_feats parameter to model call
                    lines[j] = lines[j].replace(")", ",\n" + indent_str + "    time_feats\n" + indent_str + ")")
                    
                    # Insert time extraction lines before model call
                    lines[i:i] = time_extract
                    found = True
                    break
        
        if found:
            new_content = '\n'.join(lines)
        else:
            print("Could not find the model call. Manual fix required.")
            return False
    
    # Write the fixed content
    try:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Successfully fixed {filepath}")
        return True
    except Exception as e:
        print(f"Error writing fixed file: {e}")
        return False

def main():
    print("Running GNN time features fix script...")
    
    # Fix run_gnn_baselines.py
    if fix_run_gnn_baselines():
        print("Fix applied successfully!")
        print("\nInstructions:")
        print("1. Run the GNN code again to verify the fix")
        print("2. If there are still issues, check gnn_flow_predictor.py to make time_feats optional")
    else:
        print("\nFix failed. Please apply the manual fix as described in manual_fix_instructions.txt")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
