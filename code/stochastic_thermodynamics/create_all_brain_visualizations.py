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

def create_brain_visualization(data, atlas_img, comparison, coefficient, output_path):
    """Create brain visualization for a specific comparison and coefficient."""
    
    print(f"Creating visualization for {comparison} - {coefficient}")
    
    # Get atlas data
    atlas_data = atlas_img.get_fdata()
    
    # Create significance map
    significance_data = np.zeros(atlas_data.shape[:3])  # Only first 3 dimensions
    
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
            
            # Find ROI voxels in atlas (use threshold for probabilistic atlas)
            roi_mask = (atlas_data[:,:,:,roi_idx] > 0.5)
            significance_data[roi_mask] = signed_significance
            
            roi_effects.append({
                'roi': roi_label,
                'p_value': p_val,
                'effect': diff,
                'direction': 'increase' if diff > 0 else 'decrease'
            })
    
    print(f"Found {significant_rois} significant ROIs")
    
    if significant_rois > 0:
        # Create image
        significance_img = image.new_img_like(atlas_img, significance_data)
        
        # Create colormap
        cmap = create_custom_colormap()
        
        # Determine colorbar range
        non_zero_data = significance_data[significance_data != 0]
        if len(non_zero_data) > 0:
            vmax = max(abs(np.min(non_zero_data)), abs(np.max(non_zero_data)))
        else:
            vmax = 2.0
        
        print(f"Value range: ±{vmax:.2f}")
        
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
        
        # Create plot
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        fig.suptitle(f'{title}\\n{subtitle}', fontsize=16, fontweight='bold')
        
        # Sagittal slices for good brain coverage
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
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, aspect=20, pad=0.02)
        cbar.set_label('-log₁₀(p) × sign(effect)', fontsize=12)
        
        # Add legend
        increases = sum(1 for effect in roi_effects if effect['direction'] == 'increase')
        decreases = sum(1 for effect in roi_effects if effect['direction'] == 'decrease')
        
        legend_text = (
            f"Red: Significant increases (Active > Sham): {increases}\\n"
            f"Blue: Significant decreases (Active < Sham): {decreases}\\n"
            f"Intensity: -log₁₀(p-value)\\n"
            f"Only p < 0.05 effects shown"
        )
        fig.text(0.02, 0.02, legend_text, fontsize=10, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved: {output_path}")
        return True, roi_effects
    
    else:
        print("No significant effects found to visualize")
        return False, []

def main():
    """Create all brain visualizations."""
    
    print("Loading paired results...")
    with open('paired_full_results.json', 'r') as f:
        data = json.load(f)
    
    print("Loading DiFuMo atlas...")
    difumo = datasets.fetch_atlas_difumo(dimension=1024, legacy_format=False)
    atlas_img = image.load_img(difumo.maps)
    
    print(f"Atlas loaded: {atlas_img.shape}")
    
    # Create output directory
    output_dir = 'brain_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define combinations
    coefficients = ['drift_0', 'drift_1', 'diffusion']
    comparisons = ['stimulation_minus_baseline', 'recovery_minus_baseline']
    
    summary_report = []
    summary_report.append("FUS-BOLD FOKKER-PLANCK ANALYSIS - BRAIN VISUALIZATION SUMMARY")
    summary_report.append("=" * 70)
    summary_report.append("")
    
    total_visualizations = 0
    
    # Generate all 6 visualizations
    for comparison in comparisons:
        for coefficient in coefficients:
            safe_name = f"{coefficient}_{comparison}"
            output_path = os.path.join(output_dir, f"{safe_name}_brain_slices.png")
            
            try:
                success, roi_effects = create_brain_visualization(data, atlas_img, comparison, coefficient, output_path)
                
                if success:
                    total_visualizations += 1
                    
                    # Add to summary
                    coeff_names = {
                        'drift_0': 'Drift Coefficient (Baseline)',
                        'drift_1': 'Drift Coefficient (Linear)', 
                        'diffusion': 'Diffusion Coefficient'
                    }
                    
                    comparison_names = {
                        'stimulation_minus_baseline': 'Stimulation vs Baseline',
                        'recovery_minus_baseline': 'Recovery vs Baseline'
                    }
                    
                    increases = sum(1 for effect in roi_effects if effect['direction'] == 'increase')
                    decreases = sum(1 for effect in roi_effects if effect['direction'] == 'decrease')
                    
                    summary_report.append(f"{coeff_names[coefficient]} - {comparison_names[comparison]}:")
                    summary_report.append(f"  Total significant ROIs: {len(roi_effects)}")
                    summary_report.append(f"  Increases (Active > Sham): {increases}")
                    summary_report.append(f"  Decreases (Active < Sham): {decreases}")
                    summary_report.append(f"  Visualization: {safe_name}_brain_slices.png")
                    summary_report.append("")
                    
            except Exception as e:
                print(f"Error creating visualization for {comparison}-{coefficient}: {e}")
                summary_report.append(f"{coefficient}_{comparison}: ERROR - {str(e)}")
                summary_report.append("")
    
    # Save summary report
    summary_path = os.path.join(output_dir, "visualization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\\n'.join(summary_report))
    
    print(f"\\nVisualization complete!")
    print(f"Created {total_visualizations} brain visualization(s)")
    print(f"Files saved to: {output_dir}")
    print(f"Summary report: {summary_path}")

if __name__ == "__main__":
    main()