#!/usr/bin/env python3

import numpy as np
import json
from nilearn import datasets, image

def debug_roi_mapping():
    """Debug the ROI to voxel mapping process."""
    
    print("Loading paired results...")
    with open('paired_full_results.json', 'r') as f:
        data = json.load(f)
    
    print("Loading DiFuMo atlas...")
    difumo = datasets.fetch_atlas_difumo(dimension=1024, legacy_format=False)
    atlas_img = image.load_img(difumo.maps)
    atlas_data = atlas_img.get_fdata()
    
    print(f"Atlas shape: {atlas_data.shape}")
    print(f"Atlas data type: {atlas_data.dtype}")
    print(f"Atlas value range: {atlas_data.min()} to {atlas_data.max()}")
    
    # Find a significant ROI to test
    comparison = 'stimulation_minus_baseline'
    coefficient = 'drift_0'
    
    significant_rois = []
    for roi_key, roi_data in data['roi_results'].items():
        roi_idx = roi_data['roi_index']
        coeff_data = roi_data['analysis_results'][comparison]['coefficients'][coefficient]
        p_val = coeff_data.get('p_value_paired', 1.0)
        diff = coeff_data.get('paired_difference_mean', 0)
        
        if p_val < 0.05:
            significant_rois.append({
                'key': roi_key,
                'idx': roi_idx,
                'label': roi_data['roi_label'],
                'p_val': p_val,
                'diff': diff
            })
    
    print(f"Found {len(significant_rois)} significant ROIs")
    
    # Test different thresholds and indexing approaches
    if significant_rois:
        test_roi = significant_rois[0]
        roi_idx = test_roi['idx']
        
        print(f"\nTesting ROI {test_roi['key']} (index {roi_idx}): {test_roi['label']}")
        print(f"p-value: {test_roi['p_val']:.4f}, effect: {test_roi['diff']:.4f}")
        
        # Try different approaches
        approaches = [
            ("4D indexing with threshold 0.5", lambda: atlas_data[:,:,:,roi_idx] > 0.5),
            ("4D indexing with threshold 0.1", lambda: atlas_data[:,:,:,roi_idx] > 0.1),
            ("4D indexing with threshold 0.0", lambda: atlas_data[:,:,:,roi_idx] > 0.0),
            ("Maximum probability approach", lambda: np.argmax(atlas_data, axis=3) == roi_idx),
        ]
        
        for name, approach in approaches:
            try:
                mask = approach()
                voxel_count = np.sum(mask)
                print(f"  {name}: {voxel_count} voxels")
                
                if voxel_count > 0:
                    # Show some coordinates
                    coords = np.where(mask)
                    print(f"    Sample coordinates: x={coords[0][:3]}, y={coords[1][:3]}, z={coords[2][:3]}")
                    
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
        
        # Check what values exist in this ROI's probability map
        roi_probs = atlas_data[:,:,:,roi_idx]
        unique_vals = np.unique(roi_probs[roi_probs > 0])
        print(f"  ROI probability values: {unique_vals[:10]} (showing first 10)")
        print(f"  Max probability: {roi_probs.max()}")
        
    # Check if atlas uses different indexing
    print(f"\nChecking atlas indexing...")
    print(f"Atlas 4th dimension size: {atlas_data.shape[3]}")
    print(f"Expected ROI count: 1024")
    
    # Test a few ROI indices to see their probability maps
    for test_idx in [0, 1, 2, 10, 34]:  # Include ROI 34 which should be significant
        roi_map = atlas_data[:,:,:,test_idx]
        voxel_count = np.sum(roi_map > 0.1)
        max_prob = roi_map.max()
        print(f"  ROI index {test_idx}: {voxel_count} voxels (threshold 0.1), max prob: {max_prob:.3f}")

if __name__ == "__main__":
    debug_roi_mapping()