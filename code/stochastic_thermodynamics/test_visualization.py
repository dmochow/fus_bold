#!/usr/bin/env python3

from visualize_brain_effects import load_difumo_atlas, create_significance_map
import json

def test_significance_map():
    print("Loading results...")
    with open('paired_full_results.json', 'r') as f:
        data = json.load(f)
    
    print("Loading atlas...")
    atlas_img, labels = load_difumo_atlas()
    
    print("Testing significance map creation...")
    try:
        sig_img, roi_values = create_significance_map(data, 'stimulation_minus_baseline', 'drift_0', atlas_img, labels)
        
        non_zero_count = 0
        for v in roi_values.values():
            if v != 0:
                non_zero_count += 1
        
        print(f"Found {non_zero_count} significant ROIs")
        
        # Check a few specific ROIs
        sample_rois = ['roi_0034', 'roi_0048', 'roi_0061']
        for roi_key in sample_rois:
            if roi_key in data['roi_results']:
                roi_data = data['roi_results'][roi_key]
                roi_idx = roi_data['roi_index']
                coeff_data = roi_data['analysis_results']['stimulation_minus_baseline']['coefficients']['drift_0']
                p_val = coeff_data.get('p_value_paired', 1.0)
                diff = coeff_data.get('paired_difference_mean', 0)
                roi_val = roi_values.get(roi_idx, 0)
                print(f"  {roi_key} (idx {roi_idx}): p={p_val:.4f}, diff={diff:.4f}, map_val={roi_val:.4f}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_significance_map()