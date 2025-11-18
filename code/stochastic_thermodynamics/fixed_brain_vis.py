#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting, image
import json
import os

def create_custom_colormap():
    """Create a custom colormap for significance visualization."""
    colors = ['#0066CC', '#4D94FF', '#B3D9FF', '#FFFFFF', '#FFB3B3', '#FF4D4D', '#CC0000']
    return LinearSegmentedColormap.from_list('significance', colors, N=256)

def create_medial_brain_visualization(data, atlas_img, comparison, coefficient, output_path):
    """Create brain visualization focused on medial regions."""
    
    print(f"Creating visualization for {comparison} - {coefficient}")
    
    # Get atlas data
    atlas_data = atlas_img.get_fdata()
    print(f"Atlas shape: {atlas_data.shape}")
    
    # Create significance map using probabilistic atlas approach
    significance_data = np.zeros(atlas_data.shape[:3])
    
    significant_rois = 0
    roi_effects = []
    
    for roi_key, roi_data in data['roi_results'].items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        coeff_data = roi_data['analysis_results'][comparison]['coefficients'][coefficient]
        p_val = coeff_data.get('p_value_paired', 1.0)
        diff = coeff_data.get('paired_difference_mean', 0)
        
        if p_val < 0.05:
            significant_rois += 1
            # Create signed significance value
            log_p = -np.log10(p_val)
            signed_significance = log_p * np.sign(diff)
            
            # Use maximum probability approach for ROI mapping
            # Each voxel belongs to the ROI with highest probability
            roi_prob_map = atlas_data[:,:,:,roi_idx]
            
            # Use lower threshold and weight by probability
            roi_mask = roi_prob_map > 0.1  # Lower threshold
            if np.sum(roi_mask) > 0:
                # Weight the significance by the probability
                significance_data[roi_mask] = signed_significance * roi_prob_map[roi_mask]
                print(f"  ROI {roi_idx} ({roi_label[:30]}...): {np.sum(roi_mask)} voxels, p={p_val:.4f}")
            
            roi_effects.append({
                'roi': roi_label,
                'p_value': p_val,
                'effect': diff,
                'direction': 'increase' if diff > 0 else 'decrease'
            })
    
    print(f"Found {significant_rois} significant ROIs")
    print(f"Significance map range: {significance_data.min():.3f} to {significance_data.max():.3f}")
    print(f"Non-zero voxels: {np.sum(significance_data != 0)}")
    
    if significant_rois > 0 and np.sum(significance_data != 0) > 0:
        # Create image
        significance_img = image.new_img_like(atlas_img, significance_data)
        
        # Create colormap
        cmap = create_custom_colormap()
        
        # Determine colorbar range from actual data
        non_zero_data = significance_data[significance_data != 0]
        if len(non_zero_data) > 0:
            vmax = max(abs(np.min(non_zero_data)), abs(np.max(non_zero_data)))
        else:
            vmax = 1.0
        
        print(f"Colorbar range: ±{vmax:.2f}")
        
        # Create coefficient and comparison names for title
        coeff_names = {
            'drift_0': 'Drift Coefficient (Baseline)',
            'drift_1': 'Drift Coefficient (Linear)', 
            'diffusion': 'Diffusion Coefficient'
        }
        
        comparison_names = {
            'stimulation_minus_baseline': 'Stimulation vs Baseline',
            'recovery_minus_baseline': 'Recovery vs Baseline'
        }
        
        title = f"{coeff_names[coefficient]} - {comparison_names[comparison]}"
        subtitle = f"{significant_rois} Significant ROIs (p < 0.05)"
        
        # Create plot focused on medial brain (subgenual ACC region)
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        fig.suptitle(f'{title}\\n{subtitle}', fontsize=16, fontweight='bold')
        
        # Medial sagittal slices (3-5mm spacing, focused on midline)
        x_coords = [-15, -10, -5, 0, 5, -12, -7, -2, 3, 8]  # Medial focus, 3-5mm spacing
        
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
                    threshold=0.001,  # Very low threshold to show effects
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
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label('-log₁₀(p) × sign(effect) × probability', fontsize=12)
        
        # Add legend
        increases = sum(1 for effect in roi_effects if effect['direction'] == 'increase')
        decreases = sum(1 for effect in roi_effects if effect['direction'] == 'decrease')
        
        legend_text = (
            f"Red: Significant increases (Active > Sham): {increases}\\n"
            f"Blue: Significant decreases (Active < Sham): {decreases}\\n"
            f"Intensity: -log₁₀(p-value) × ROI probability\\n"
            f"Focus: Medial brain (subgenual ACC target)"
        )
        fig.text(0.02, 0.02, legend_text, fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        return True, roi_effects
    
    else:
        print("No significant effects found to visualize or no voxels mapped")
        return False, []

def main():
    """Create focused medial brain visualizations."""
    
    print("Loading paired results...")
    with open('paired_full_results.json', 'r') as f:
        data = json.load(f)
    
    print("Loading DiFuMo atlas...")
    difumo = datasets.fetch_atlas_difumo(dimension=1024, legacy_format=False)
    atlas_img = image.load_img(difumo.maps)
    
    print(f"Atlas loaded: {atlas_img.shape}")
    
    # Create output directory
    output_dir = 'medial_brain_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Test with one visualization first
    comparison = 'stimulation_minus_baseline'
    coefficient = 'drift_0'
    
    output_path = os.path.join(output_dir, f"test_{coefficient}_{comparison}_medial.png")
    success, roi_effects = create_medial_brain_visualization(data, atlas_img, comparison, coefficient, output_path)
    
    if success:
        print("\\nTest visualization successful! Proceeding with all combinations...")
        
        # Generate all 6 visualizations
        coefficients = ['drift_0', 'drift_1', 'diffusion']
        comparisons = ['stimulation_minus_baseline', 'recovery_minus_baseline']
        
        for comp in comparisons:
            for coeff in coefficients:
                if comp == comparison and coeff == coefficient:
                    continue  # Already done
                    
                output_path = os.path.join(output_dir, f"{coeff}_{comp}_medial.png")
                success, _ = create_medial_brain_visualization(data, atlas_img, comp, coeff, output_path)
    
    print(f"\\nVisualization complete! Files saved to: {output_dir}")

if __name__ == "__main__":
    main()