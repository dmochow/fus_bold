#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting, image
import json
import os

def create_simple_visualization():
    """Create a simple brain visualization for one condition."""
    
    print("Loading results...")
    with open('paired_full_results.json', 'r') as f:
        data = json.load(f)
    
    print("Loading DiFuMo atlas...")
    difumo = datasets.fetch_atlas_difumo(dimension=1024, legacy_format=False)
    atlas_img = image.load_img(difumo.maps)
    labels = difumo.labels['difumo_names'].tolist()
    
    print(f"Atlas loaded: {atlas_img.shape}")
    
    # Create significance map for stimulation vs baseline, drift_0
    comparison = 'stimulation_minus_baseline'
    coefficient = 'drift_0'
    
    print(f"Creating map for {comparison} - {coefficient}")
    
    # Get atlas data
    atlas_data = atlas_img.get_fdata()
    
    # Create significance map
    significance_data = np.zeros(atlas_data.shape[:3])  # Only first 3 dimensions
    
    significant_rois = 0
    for roi_key, roi_data in data['roi_results'].items():
        roi_idx = roi_data['roi_index']
        coeff_data = roi_data['analysis_results'][comparison]['coefficients'][coefficient]
        p_val = coeff_data.get('p_value_paired', 1.0)
        diff = coeff_data.get('paired_difference_mean', 0)
        
        if p_val < 0.05:
            significant_rois += 1
            # Create signed significance value
            log_p = -np.log10(p_val)
            signed_significance = log_p * np.sign(diff)
            
            # Find ROI voxels in atlas (ROI indices are 1-based in DiFuMo)
            roi_mask = (atlas_data[:,:,:,roi_idx] > 0.5)  # Use threshold for probabilistic atlas
            significance_data[roi_mask] = signed_significance
    
    print(f"Found {significant_rois} significant ROIs")
    
    if significant_rois > 0:
        # Create image
        significance_img = image.new_img_like(atlas_img, significance_data)
        
        # Create colormap
        colors = ['#0066CC', '#4D94FF', '#B3D9FF', '#FFFFFF', '#FFB3B3', '#FF4D4D', '#CC0000']
        cmap = LinearSegmentedColormap.from_list('significance', colors, N=256)
        
        # Determine colorbar range
        non_zero_data = significance_data[significance_data != 0]
        if len(non_zero_data) > 0:
            vmax = max(abs(np.min(non_zero_data)), abs(np.max(non_zero_data)))
        else:
            vmax = 2.0
        
        print(f"Value range: ±{vmax:.2f}")
        
        # Create plot
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Drift Coefficient (Baseline) - Stimulation vs Baseline\n{significant_rois} Significant ROIs', 
                     fontsize=16, fontweight='bold')
        
        # Sagittal slices
        x_coords = [-50, -30, -10, 10, 30, -40, -20, 0, 20, 40]
        
        for i, x_coord in enumerate(x_coords):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            try:
                plotting.plot_stat_map(
                    significance_img,
                    cut_coords=[x_coord],
                    display_mode='x',
                    figure=fig,
                    axes=ax,
                    colorbar=False,
                    cmap=cmap,
                    symmetric_cbar=True,
                    vmax=vmax,
                    threshold=0.01,
                    title=f'x = {x_coord}',
                    annotate=False
                )
            except Exception as e:
                print(f"Warning: Could not plot slice at x={x_coord}: {e}")
                ax.text(0.5, 0.5, f'x = {x_coord}\\n(no data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, aspect=20)
        cbar.set_label('-log₁₀(p) × sign(effect)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('test_brain_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Saved test_brain_visualization.png")
    
    else:
        print("No significant effects found to visualize")

if __name__ == "__main__":
    create_simple_visualization()