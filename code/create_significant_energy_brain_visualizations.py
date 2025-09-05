#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting, image
import nibabel as nib
import os
import warnings
warnings.filterwarnings('ignore')

def create_custom_colormap():
    """Create a custom colormap for energy change visualization."""
    # Blue for decreases, white for no change, red for increases
    colors = ['#0066CC', '#4D94FF', '#B3D9FF', '#FFFFFF', '#FFB3B3', '#FF4D4D', '#CC0000']
    return LinearSegmentedColormap.from_list('energy_changes', colors, N=256)

def load_energy_statistics():
    """Load the component statistics with significance testing."""
    
    print("Loading component statistics...")
    
    try:
        stats_df = pd.read_csv('free_energy_component_statistics.csv')
        print(f"Loaded statistics for {len(stats_df)} ROI-comparison-component combinations")
        return stats_df
        
    except FileNotFoundError:
        print("Error: free_energy_component_statistics.csv not found")
        return None

def load_energy_changes():
    """Load the subject-averaged energy changes."""
    
    print("Loading energy change data...")
    
    try:
        # Load component changes data
        changes_df = pd.read_csv('free_energy_component_changes.csv')
        
        # Focus on active condition and compute subject averages
        active_changes = changes_df[changes_df['condition'] == 'active']
        
        # Group by ROI and comparison, then average across subjects
        roi_averages = active_changes.groupby(['roi_idx', 'comparison']).agg({
            'free_energy_change': 'mean',
            'potential_energy_change': 'mean',
            'entropy_change': 'mean',
            'roi_label': 'first'
        }).reset_index()
        
        print(f"Loaded energy changes for {len(roi_averages)} ROI-comparison combinations")
        return roi_averages
        
    except FileNotFoundError:
        print("Error: free_energy_component_changes.csv not found")
        return None

def get_significant_rois(stats_df, metric, comparison, alpha=0.05):
    """Get ROIs that are statistically significant for a given metric and comparison."""
    
    # Filter for the specific metric and comparison
    filtered_stats = stats_df[
        (stats_df['component'] == metric) & 
        (stats_df['comparison'] == comparison) &
        (stats_df['p_value'] < alpha)
    ]
    
    print(f"Found {len(filtered_stats)} significant ROIs for {metric} {comparison} (p < {alpha})")
    
    return filtered_stats

def load_difumo_atlas():
    """Load the DiFuMo atlas."""
    
    print("Loading DiFuMo atlas...")
    
    try:
        # Load DiFuMo 1024 atlas
        difumo = datasets.fetch_atlas_difumo(dimension=1024, legacy_format=False)
        atlas_img = image.load_img(difumo.maps)
        
        print(f"Atlas loaded: {atlas_img.shape}")
        return atlas_img
        
    except Exception as e:
        print(f"Error loading DiFuMo atlas: {e}")
        return None

def create_significant_energy_map(roi_averages, stats_df, atlas_img, metric='free_energy_change', 
                                 comparison='stimulation_minus_baseline', alpha=0.05):
    """Create a brain map showing only statistically significant energy changes."""
    
    print(f"Creating significant {metric} map for {comparison}...")
    
    # Get significant ROIs
    sig_stats = get_significant_rois(stats_df, metric, comparison, alpha)
    
    if len(sig_stats) == 0:
        print(f"No significant ROIs found for {metric} {comparison}")
        return None, []
    
    # Get the significant ROI indices
    sig_roi_indices = set(sig_stats['roi_idx'].values)
    
    # Filter energy changes for significant ROIs only
    filtered_data = roi_averages[
        (roi_averages['comparison'] == comparison) &
        (roi_averages['roi_idx'].isin(sig_roi_indices))
    ]
    
    print(f"Found energy change data for {len(filtered_data)} significant ROIs")
    
    # Get atlas data
    atlas_data = atlas_img.get_fdata()
    
    # Create energy map (only for significant ROIs)
    energy_map = np.zeros(atlas_data.shape[:3])  # Only first 3 dimensions
    
    # Track ROI information
    roi_info = []
    
    # Map energy values to brain regions (only significant ROIs)
    for _, row in filtered_data.iterrows():
        roi_idx = int(row['roi_idx'])
        energy_value = row[metric]
        roi_label = row['roi_label']
        
        # Get corresponding statistics
        roi_stats = sig_stats[sig_stats['roi_idx'] == roi_idx]
        if len(roi_stats) > 0:
            p_value = roi_stats.iloc[0]['p_value']
            cohens_d = roi_stats.iloc[0]['cohens_d']
        else:
            continue
        
        # DiFuMo is 4D with the 4th dimension being ROI index
        if len(atlas_data.shape) == 4:
            roi_map = atlas_data[:, :, :, roi_idx]
        else:
            roi_map = atlas_data
        
        # Find voxels corresponding to this ROI using threshold
        roi_mask = roi_map > 0.001
        
        if np.any(roi_mask):
            energy_map[roi_mask] = energy_value
            roi_info.append({
                'roi_idx': roi_idx,
                'roi_label': roi_label,
                'energy_value': energy_value,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_voxels': np.sum(roi_mask)
            })
        
    print(f"Mapped {len(roi_info)} significant ROIs to brain voxels")
    
    # Create nibabel image
    energy_img = nib.Nifti1Image(energy_map, atlas_img.affine)
    
    return energy_img, roi_info

def create_significant_brain_visualization(roi_averages, stats_df, atlas_img, metric, comparison, output_path, alpha=0.05):
    """Create brain visualization for significant energy changes only."""
    
    print(f"\\nCreating significant-only visualization for {metric} - {comparison}")
    
    # Create energy map for significant ROIs only
    energy_img, roi_info = create_significant_energy_map(
        roi_averages, stats_df, atlas_img, metric, comparison, alpha
    )
    
    if energy_img is None or len(roi_info) == 0:
        print("No significant ROIs to visualize")
        return False
    
    # Get data statistics
    energy_data = energy_img.get_fdata()
    nonzero_data = energy_data[energy_data != 0]
    
    if len(nonzero_data) == 0:
        print("No non-zero energy values found")
        return False
    
    # Calculate range
    vmin, vmax = np.min(nonzero_data), np.max(nonzero_data)
    
    # Make colorbar symmetric around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    print(f"Significant energy range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"Significant voxels: {len(nonzero_data)}")
    
    # Create titles
    metric_names = {
        'free_energy_change': 'Free Energy Change',
        'potential_energy_change': 'Potential Energy Change',
        'entropy_change': 'Entropy Change'
    }
    
    comparison_names = {
        'stimulation_minus_baseline': 'During Active Stimulation',
        'recovery_minus_baseline': 'After Active Stimulation'
    }
    
    title = f"{metric_names.get(metric, metric)} - {comparison_names.get(comparison, comparison)}"
    subtitle = f"Significant Effects Only (p < {alpha}) - {len(roi_info)} ROIs"
    
    # Create colormap
    cmap = create_custom_colormap()
    
    # Create figure with sagittal slices focused on medial regions
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle(f'{title}\\n{subtitle}', fontsize=16, fontweight='bold')
    
    # Focus on medial sagittal slices where subgenual ACC target is located
    x_coords = [-15, -10, -5, 0, 5, -8, -4, -2, 2, 4]
    
    for i, x_coord in enumerate(x_coords):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        try:
            plotting.plot_stat_map(
                energy_img,
                cut_coords=[x_coord],
                display_mode='x',
                figure=fig,
                axes=ax,
                colorbar=False,
                cmap=cmap,
                symmetric_cbar=True,
                vmax=abs_max,
                threshold=0.001,  # Small threshold to show all significant effects
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('Significant Energy Change', fontsize=12)
    
    # Add statistics
    positive_changes = np.sum(nonzero_data > 0)
    negative_changes = np.sum(nonzero_data < 0)
    
    # Calculate p-value statistics
    p_values = [roi['p_value'] for roi in roi_info]
    effect_sizes = [roi['cohens_d'] for roi in roi_info]
    
    stats_text = (
        f"Red: Increases ({positive_changes:,} voxels)\\n"
        f"Blue: Decreases ({negative_changes:,} voxels)\\n"
        f"Significant ROIs: {len(roi_info)}\\n"
        f"Mean p-value: {np.mean(p_values):.4f}\\n"
        f"Mean effect size: {np.mean(np.abs(effect_sizes)):.3f}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return True

def main():
    """Create brain visualizations for significant energy changes only."""
    
    print("=== SIGNIFICANT ENERGY BRAIN VISUALIZATIONS ===")
    print("Creating sagittal brain overlays showing only statistically significant changes")
    print()
    
    # Load data
    stats_df = load_energy_statistics()
    if stats_df is None:
        return
        
    roi_averages = load_energy_changes()
    if roi_averages is None:
        return
    
    # Load atlas
    atlas_img = load_difumo_atlas()
    if atlas_img is None:
        return
    
    # Create output directory
    output_dir = 'significant_energy_brain_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define visualizations to create
    metrics = ['free_energy_change', 'potential_energy_change']
    comparisons = ['stimulation_minus_baseline', 'recovery_minus_baseline']
    alpha = 0.05
    
    visualizations_created = 0
    summary_data = []
    
    # Create each visualization
    for metric in metrics:
        for comparison in comparisons:
            
            # Create safe filename
            safe_name = f"{metric}_{comparison}_significant"
            output_path = os.path.join(output_dir, f"{safe_name}_brain_overlay.png")
            
            try:
                success = create_significant_brain_visualization(
                    roi_averages, stats_df, atlas_img, metric, comparison, output_path, alpha
                )
                
                if success:
                    visualizations_created += 1
                    
                    # Get significance stats for summary
                    sig_stats = get_significant_rois(stats_df, metric, comparison, alpha)
                    increases = np.sum(sig_stats['cohens_d'] > 0)
                    decreases = np.sum(sig_stats['cohens_d'] < 0)
                    
                    summary_data.append({
                        'metric': metric,
                        'comparison': comparison,
                        'significant_rois': len(sig_stats),
                        'increases': increases,
                        'decreases': decreases,
                        'filename': f"{safe_name}_brain_overlay.png"
                    })
                    
            except Exception as e:
                print(f"Error creating visualization for {metric}-{comparison}: {e}")
    
    # Create summary
    summary_lines = [
        "FUS-BOLD FREE ENERGY ANALYSIS - SIGNIFICANT EFFECTS BRAIN VISUALIZATION",
        "=" * 70,
        "",
        f"Significance threshold: p < {alpha}",
        f"Visualizations created: {visualizations_created}/4",
        "",
        "Significant effects summary:",
    ]
    
    for data in summary_data:
        metric_display = data['metric'].replace('_', ' ').title()
        comparison_display = "During Stimulation" if 'stimulation' in data['comparison'] else "After Stimulation"
        
        summary_lines.extend([
            f"",
            f"{metric_display} {comparison_display}:",
            f"  File: {data['filename']}",
            f"  Significant ROIs: {data['significant_rois']}",
            f"  Increases: {data['increases']} ROIs",
            f"  Decreases: {data['decreases']} ROIs"
        ])
    
    summary_lines.extend([
        "",
        "Color scale:",
        "  - Red: Significant energy increases (Active > Sham)", 
        "  - Blue: Significant energy decreases (Active < Sham)",
        "  - Gray background: Non-significant regions (not shown)",
        "",
        "Focus: Medial sagittal slices (subgenual ACC target region)",
        "Analysis: Paired t-tests comparing active vs sham conditions"
    ])
    
    # Save summary
    summary_path = os.path.join(output_dir, "significant_visualization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\\n'.join(summary_lines))
    
    print(f"\\n=== SIGNIFICANT VISUALIZATION COMPLETE ===")
    print(f"Created {visualizations_created} brain visualization(s)")
    print(f"Files saved to: {output_dir}/")
    print(f"Summary: {summary_path}")
    
    # Display significance findings
    print(f"\\n=== SIGNIFICANCE SUMMARY ===")
    total_sig_rois = 0
    for data in summary_data:
        metric_name = data['metric'].replace('_', ' ').title()
        comp_name = "During" if 'stimulation' in data['comparison'] else "After"
        
        print(f"{metric_name} {comp_name}: {data['significant_rois']} significant ROIs")
        print(f"  Increases: {data['increases']}, Decreases: {data['decreases']}")
        total_sig_rois += data['significant_rois']
    
    print(f"\\nTotal significant effects across all analyses: {total_sig_rois}")

if __name__ == "__main__":
    main()