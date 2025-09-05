#\!/usr/bin/env python3

import json

def test_data_loading():
    print("Loading paired results...")
    with open('paired_full_results.json', 'r') as f:
        data = json.load(f)
    
    # Test one specific ROI that should be significant
    roi_key = 'roi_0034'  # This should have p=0.0437 for stimulation drift_0
    roi_data = data['roi_results'][roi_key]
    roi_idx = roi_data['roi_index']
    
    print(f"Testing ROI {roi_key} (index {roi_idx}):")
    
    # Check stimulation vs baseline, drift_0
    coeff_data = roi_data['analysis_results']['stimulation_minus_baseline']['coefficients']['drift_0']
    p_val = coeff_data.get('p_value_paired', 1.0)
    diff = coeff_data.get('paired_difference_mean', 0)
    
    print(f"  p_value_paired: {p_val}")
    print(f"  paired_difference_mean: {diff}")
    print(f"  Should be significant: {p_val < 0.05}")
    
    # Count all significant effects
    total_significant = 0
    for roi_k, roi_d in data['roi_results'].items():
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                cd = roi_d['analysis_results'][comparison]['coefficients'][coeff]
                if cd.get('p_value_paired', 1.0) < 0.05:
                    total_significant += 1
    
    print(f"\nTotal significant effects: {total_significant}")
    return True

if __name__ == "__main__":
    test_data_loading()
EOF < /dev/null