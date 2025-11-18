import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting, image
from nilearn.plotting import plot_stat_map, plot_glass_brain
import json
import argparse
import os

def load_difumo_atlas():
    """Load DiFuMo atlas and return maps and labels."""
    print("Loading DiFuMo atlas...")
    from nilearn import image
    
    # Fetch atlas data
    difumo = datasets.fetch_atlas_difumo(dimension=1024, legacy_format=False)
    
    # Load the actual image from the file path
    if isinstance(difumo.maps, str):
        # maps is a file path, load it as an image
        atlas_img = image.load_img(difumo.maps)
    else:
        # maps is already an image object
        atlas_img = difumo.maps
    
    # Process labels
    if hasattr(difumo.labels, 'to_numpy'):
        # DataFrame format (new)
        labels = difumo.labels['difumo_names'].tolist()
    else:
        # Legacy format
        labels = [str(label) for label in difumo.labels]
    
    print(f"Loaded DiFuMo atlas with {len(labels)} ROIs")
    print(f"Atlas image type: {type(atlas_img)}")
    print(f"Atlas image shape: {atlas_img.shape}")
    return atlas_img, labels

def create_significance_map(baseline_corrected_results, comparison, coefficient, atlas_img, labels):
    """Create a 3D significance map from statistical results."""
    
    print(f"Creating significance map for {comparison} - {coefficient}")
    
    # Extract significance values for this comparison and coefficient
    roi_values = {}  # roi_index -> significance value
    
    for roi_key, roi_data in baseline_corrected_results['roi_results'].items():
        roi_idx = roi_data['roi_index']
        
        # Handle different JSON formats
        if 'baseline_corrected_comparisons' in roi_data['analysis_results']:
            # Original format
            comparison_data = roi_data['analysis_results']['baseline_corrected_comparisons'][comparison]
            coeff_data = comparison_data['coefficients'][coefficient]
            p_val = coeff_data.get('p_value_active_vs_sham', 1.0)
            change_diff = coeff_data.get('change_difference', 0)
        else:
            # Paired format - comparisons are at top level
            comparison_data = roi_data['analysis_results'][comparison]
            coeff_data = comparison_data['coefficients'][coefficient]
            p_val = coeff_data.get('p_value_paired', 1.0)
            change_diff = coeff_data.get('paired_difference_mean', 0)
        
        # Only include significant effects (p < 0.05)
        if p_val < 0.05:
            # Create signed significance value
            # Positive for increases, negative for decreases
            # Magnitude reflects -log10(p_value)
            log_p = -np.log10(p_val)
            signed_significance = log_p * np.sign(change_diff)
            roi_values[roi_idx] = signed_significance
        else:
            roi_values[roi_idx] = 0
    
    # Create the 3D map  
    atlas_data = atlas_img.get_fdata()
    significance_data = np.zeros(atlas_data.shape[:3])  # Only first 3 dimensions
    
    # Map ROI values to brain voxels
    for roi_idx, sig_value in roi_values.items():
        if sig_value != 0:  # Only non-zero values
            # DiFuMo atlas has ROI data in 4th dimension, use probabilistic threshold
            roi_mask = (atlas_data[:,:,:,roi_idx] > 0.5)
            significance_data[roi_mask] = sig_value
    
    # Create new image
    significance_img = image.new_img_like(atlas_img, significance_data)
    
    significant_count = len([v for v in roi_values.values() if v != 0])
    print(f"Created significance map with {significant_count} significant ROIs")
    
    if significant_count > 0:
        non_zero_data = significance_data[significance_data != 0]
        print(f"Value range: {np.min(non_zero_data):.2f} to {np.max(non_zero_data):.2f}")
    else:
        print("No significant effects found")
    
    return significance_img, roi_values

def create_custom_colormap():
    """Create a custom colormap for significance visualization."""
    
    # Blue (decreases) to white to red (increases)
    colors = ['#0066CC', '#4D94FF', '#B3D9FF', '#FFFFFF', '#FFB3B3', '#FF4D4D', '#CC0000']
    n_bins = 256
    
    cmap = LinearSegmentedColormap.from_list('significance', colors, N=n_bins)
    return cmap

def plot_brain_slices(significance_img, title, output_path, vmax=None):
    """Plot brain slices showing significant effects."""
    
    if vmax is None:
        # Auto-determine colorbar range
        data = significance_img.get_fdata()
        non_zero_data = data[data != 0]
        if len(non_zero_data) > 0:
            vmax = max(abs(np.min(non_zero_data)), abs(np.max(non_zero_data)))
        else:
            vmax = 1.0
    
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Define slice coordinates for good brain coverage
    # Sagittal slices from left to right
    sagittal_coords = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40]
    
    # Create 2x5 subplot grid
    for i, x_coord in enumerate(sagittal_coords):
        ax = plt.subplot(2, 5, i + 1)
        
        try:
            # Plot sagittal slice
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
                threshold=0.01,  # Only show non-zero values
                title=f'x = {x_coord}',
                annotate=False
            )
            
        except Exception as e:
            print(f"Warning: Could not plot slice at x={x_coord}: {e}")
            ax.text(0.5, 0.5, f'x = {x_coord}\n(no data)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=fig.get_axes(), shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('-log₁₀(p) × sign(effect)', fontsize=12)
    
    # Add legend
    legend_text = (
        "Red: Significant increases (Active > Sham)\n"
        "Blue: Significant decreases (Active < Sham)\n"
        "Intensity: -log₁₀(p-value)\n"
        "Only p < 0.05 effects shown"
    )
    fig.text(0.02, 0.02, legend_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved brain visualization: {output_path}")

def create_glass_brain_plot(significance_img, title, output_path, vmax=None):
    """Create a glass brain plot for overview."""
    
    if vmax is None:
        data = significance_img.get_fdata()
        non_zero_data = data[data != 0]
        if len(non_zero_data) > 0:
            vmax = max(abs(np.min(non_zero_data)), abs(np.max(non_zero_data)))
        else:
            vmax = 1.0
    
    cmap = create_custom_colormap()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title + " - Glass Brain View", fontsize=14, fontweight='bold')
    
    # Plot three orthogonal views
    views = ['x', 'y', 'z']
    view_names = ['Sagittal', 'Coronal', 'Axial']
    
    for i, (view, view_name) in enumerate(zip(views, view_names)):
        try:
            plotting.plot_glass_brain(
                significance_img,
                figure=fig,
                axes=axes[i],
                display_mode=view,
                colorbar=False,
                cmap=cmap,
                symmetric_cbar=True,
                vmax=vmax,
                threshold=0.01,
                title=view_name
            )
        except Exception as e:
            print(f"Warning: Could not create glass brain {view} view: {e}")
            axes[i].text(0.5, 0.5, f'{view_name}\n(no data)', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, shrink=0.8, aspect=20)
    cbar.set_label('-log₁₀(p) × sign(effect)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved glass brain plot: {output_path}")

def generate_summary_statistics(baseline_corrected_results, comparison, coefficient):
    """Generate summary statistics for the visualization."""
    
    stats = {
        'total_rois': 0,
        'significant_rois': 0,
        'increases': 0,
        'decreases': 0,
        'max_increase_p': 1.0,
        'max_decrease_p': 1.0,
        'max_increase_roi': None,
        'max_decrease_roi': None
    }
    
    max_increase_val = 0
    max_decrease_val = 0
    
    for roi_key, roi_data in baseline_corrected_results['roi_results'].items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        stats['total_rois'] += 1
        
        # Handle different JSON formats
        if 'baseline_corrected_comparisons' in roi_data['analysis_results']:
            # Original format
            comparison_data = roi_data['analysis_results']['baseline_corrected_comparisons'][comparison]
            coeff_data = comparison_data['coefficients'][coefficient]
            p_val = coeff_data.get('p_value_active_vs_sham', 1.0)
            change_diff = coeff_data.get('change_difference', 0)
        else:
            # Paired format
            comparison_data = roi_data['analysis_results'][comparison]
            coeff_data = comparison_data['coefficients'][coefficient]
            p_val = coeff_data.get('p_value_paired', 1.0)
            change_diff = coeff_data.get('paired_difference_mean', 0)
        
        if p_val < 0.05:
            stats['significant_rois'] += 1
            
            if change_diff > 0:  # Increase
                stats['increases'] += 1
                if change_diff > max_increase_val:
                    max_increase_val = change_diff
                    stats['max_increase_p'] = p_val
                    stats['max_increase_roi'] = f"ROI {roi_idx}: {roi_label[:40]}..."
                    
            elif change_diff < 0:  # Decrease
                stats['decreases'] += 1
                if abs(change_diff) > max_decrease_val:
                    max_decrease_val = abs(change_diff)
                    stats['max_decrease_p'] = p_val
                    stats['max_decrease_roi'] = f"ROI {roi_idx}: {roi_label[:40]}..."
    
    return stats

def visualize_all_effects(json_file, output_dir):
    """Create visualizations for all coefficient-comparison combinations."""
    
    print(f"Loading results from {json_file}...")
    with open(json_file, 'r') as f:
        baseline_corrected_results = json.load(f)
    
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load atlas
    atlas_img, labels = load_difumo_atlas()
    
    # Define combinations
    coefficients = ['drift_0', 'drift_1', 'diffusion']
    comparisons = ['stimulation_minus_baseline', 'recovery_minus_baseline']
    
    coeff_names = {
        'drift_0': 'Drift Coefficient (Baseline)',
        'drift_1': 'Drift Coefficient (Linear)', 
        'diffusion': 'Diffusion Coefficient'
    }
    
    comparison_names = {
        'stimulation_minus_baseline': 'Stimulation vs Baseline',
        'recovery_minus_baseline': 'Recovery vs Baseline'
    }
    
    # Determine global colorbar range for consistency
    all_values = []
    for comparison in comparisons:
        for coefficient in coefficients:
            try:
                _, roi_values = create_significance_map(
                    baseline_corrected_results, comparison, coefficient, atlas_img, labels
                )
                values = [v for v in roi_values.values() if v != 0]
                all_values.extend(values)
            except Exception as e:
                print(f"Warning: Could not process {comparison}-{coefficient}: {e}")
    
    if all_values:
        global_vmax = max(abs(min(all_values)), abs(max(all_values)))
    else:
        global_vmax = 2.0  # Default value
    
    print(f"Using global colorbar range: ±{global_vmax:.2f}")
    
    # Create summary report
    summary_report = []
    summary_report.append("FUS-BOLD FOKKER-PLANCK ANALYSIS - BRAIN VISUALIZATION SUMMARY")
    summary_report.append("=" * 70)
    summary_report.append("")
    
    # Generate visualizations
    for comparison in comparisons:
        for coefficient in coefficients:
            print(f"\nProcessing {comparison} - {coefficient}...")
            
            try:
                # Create significance map
                significance_img, roi_values = create_significance_map(
                    baseline_corrected_results, comparison, coefficient, atlas_img, labels
                )
                
                # Generate statistics
                stats = generate_summary_statistics(
                    baseline_corrected_results, comparison, coefficient
                )
                
                # Create title
                title = f"{coeff_names[coefficient]} - {comparison_names[comparison]}"
                
                # Create file names
                safe_name = f"{coefficient}_{comparison}"
                slices_output = os.path.join(output_dir, f"{safe_name}_brain_slices.png")
                glass_output = os.path.join(output_dir, f"{safe_name}_glass_brain.png")
                
                # Create visualizations
                plot_brain_slices(significance_img, title, slices_output, global_vmax)
                create_glass_brain_plot(significance_img, title, glass_output, global_vmax)
                
                # Add to summary report
                summary_report.append(f"{title}:")
                summary_report.append(f"  Total ROIs: {stats['total_rois']}")
                summary_report.append(f"  Significant ROIs: {stats['significant_rois']}")
                summary_report.append(f"  Increases (Active > Sham): {stats['increases']}")
                summary_report.append(f"  Decreases (Active < Sham): {stats['decreases']}")
                if stats['max_increase_roi']:
                    summary_report.append(f"  Strongest increase: {stats['max_increase_roi']} (p={stats['max_increase_p']:.4f})")
                if stats['max_decrease_roi']:
                    summary_report.append(f"  Strongest decrease: {stats['max_decrease_roi']} (p={stats['max_decrease_p']:.4f})")
                summary_report.append("")
                
            except Exception as e:
                print(f"Error processing {comparison}-{coefficient}: {e}")
                title = f"{coeff_names[coefficient]} - {comparison_names[comparison]}"
                summary_report.append(f"{title}: ERROR - {str(e)}")
                summary_report.append("")
    
    # Save summary report
    summary_path = os.path.join(output_dir, "visualization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_report))
    
    print(f"\nVisualization complete!")
    print(f"Files saved to: {output_dir}")
    print(f"Summary report: {summary_path}")
    
    # Print quick summary
    total_significant = sum([len([v for v in roi_vals.values() if v != 0]) 
                           for roi_vals in [create_significance_map(baseline_corrected_results, comp, coeff, atlas_img, labels)[1] 
                                          for comp in comparisons for coeff in coefficients]])
    print(f"Total significant effects visualized: {total_significant}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize FUS-BOLD Fokker-Planck brain effects')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with baseline-corrected results')
    parser.add_argument('--output-dir', type=str, default='brain_visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    json_file = args.input
    output_dir = os.path.join('/Users/jacekdmochowski/PROJECTS/fus_bold/code', args.output_dir)
    
    try:
        visualize_all_effects(json_file, output_dir)
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()