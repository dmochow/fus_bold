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

def create_energy_map(roi_averages, atlas_img, metric='free_energy_change', 
                     comparison='stimulation_minus_baseline'):
    """Create a brain map showing energy changes."""
    
    print(f"Creating {metric} map for {comparison}...")
    
    # Filter data for the specific comparison
    filtered_data = roi_averages[roi_averages['comparison'] == comparison]
    
    if len(filtered_data) == 0:
        print(f"No data found for {comparison}")
        return None, []
    
    print(f"Found data for {len(filtered_data)} ROIs")
    
    # Get atlas data
    atlas_data = atlas_img.get_fdata()
    
    # Create energy map
    energy_map = np.zeros(atlas_data.shape[:3])  # Only first 3 dimensions
    
    # Track ROI information
    roi_info = []
    
    # Map energy values to brain regions
    for _, row in filtered_data.iterrows():
        roi_idx = int(row['roi_idx'])
        energy_value = row[metric]
        roi_label = row['roi_label']
        
        # DiFuMo is 4D with the 4th dimension being ROI index
        if len(atlas_data.shape) == 4:
            roi_map = atlas_data[:, :, :, roi_idx]
        else:
            roi_map = atlas_data
        
        # Find voxels corresponding to this ROI using threshold
        # DiFuMo uses probabilistic values, so threshold at small positive value
        roi_mask = roi_map > 0.001
        
        if np.any(roi_mask):
            energy_map[roi_mask] = energy_value
            roi_info.append({
                'roi_idx': roi_idx,
                'roi_label': roi_label,
                'energy_value': energy_value,
                'n_voxels': np.sum(roi_mask)
            })
        
    print(f"Mapped {len(roi_info)} ROIs to brain voxels")
    
    # Create nibabel image
    energy_img = nib.Nifti1Image(energy_map, atlas_img.affine)
    
    return energy_img, roi_info

def create_brain_visualization(roi_averages, atlas_img, metric, comparison, output_path):
    """Create brain visualization for energy changes."""
    
    print(f"\\nCreating visualization for {metric} - {comparison}")
    
    # Create energy map
    energy_img, roi_info = create_energy_map(roi_averages, atlas_img, metric, comparison)
    
    if energy_img is None:
        print("Could not create energy map")
        return False
    
    # Get data statistics
    energy_data = energy_img.get_fdata()
    nonzero_data = energy_data[energy_data != 0]
    
    if len(nonzero_data) == 0:
        print("No non-zero energy values found")
        return False
    
    # Calculate percentile-based range to avoid outliers
    vmin, vmax = np.percentile(nonzero_data, [5, 95])
    
    # Make colorbar symmetric around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    print(f"Energy range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"Non-zero voxels: {len(nonzero_data)}")
    
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
    subtitle = f"Subject-Averaged Changes (Active vs Baseline)"
    
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
                threshold=abs(vmin) * 0.1,  # Small threshold to reduce noise
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
    cbar.set_label('Energy Change (Subject Average)', fontsize=12)
    
    # Add statistics
    positive_changes = np.sum(nonzero_data > 0)
    negative_changes = np.sum(nonzero_data < 0)
    
    stats_text = (
        f"Red: Increases ({positive_changes:,} voxels)\\n"
        f"Blue: Decreases ({negative_changes:,} voxels)\\n"
        f"Mean change: {np.mean(nonzero_data):.4f}\\n"
        f"ROIs mapped: {len(roi_info)}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return True

def main():
    """Create brain visualizations for energy changes."""
    
    print("=== ENERGY BRAIN VISUALIZATIONS ===")
    print("Creating sagittal brain overlays of free energy and potential energy changes")
    print()
    
    # Load data
    roi_averages = load_energy_changes()
    if roi_averages is None:
        return
    
    # Load atlas
    atlas_img = load_difumo_atlas()
    if atlas_img is None:
        return
    
    # Create output directory
    output_dir = 'energy_brain_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define visualizations to create
    metrics = ['free_energy_change', 'potential_energy_change']
    comparisons = ['stimulation_minus_baseline', 'recovery_minus_baseline']
    
    visualizations_created = 0
    
    # Create each visualization
    for metric in metrics:
        for comparison in comparisons:
            
            # Create safe filename
            safe_name = f"{metric}_{comparison}"
            output_path = os.path.join(output_dir, f"{safe_name}_brain_overlay.png")
            
            try:
                success = create_brain_visualization(
                    roi_averages, atlas_img, metric, comparison, output_path
                )
                
                if success:
                    visualizations_created += 1
                    
            except Exception as e:
                print(f"Error creating visualization for {metric}-{comparison}: {e}")
    
    # Create summary
    summary_lines = [
        "FUS-BOLD FREE ENERGY ANALYSIS - BRAIN VISUALIZATION SUMMARY",
        "=" * 60,
        "",
        f"Visualizations created: {visualizations_created}/4",
        "",
        "Files created:",
    ]
    
    for metric in metrics:
        for comparison in comparisons:
            safe_name = f"{metric}_{comparison}"
            filename = f"{safe_name}_brain_overlay.png"
            
            metric_display = metric.replace('_', ' ').title()
            comparison_display = "During Stimulation" if 'stimulation' in comparison else "After Stimulation"
            
            summary_lines.append(f"  - {filename}: {metric_display} {comparison_display}")
    
    summary_lines.extend([
        "",
        "Color scale:",
        "  - Red: Energy increases (Active > Baseline)", 
        "  - Blue: Energy decreases (Active < Baseline)",
        "  - White: No change",
        "",
        "Focus: Medial sagittal slices (subgenual ACC target region)",
        "Data: Subject-averaged changes across all 1024 DiFuMo ROIs"
    ])
    
    # Save summary
    summary_path = os.path.join(output_dir, "visualization_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\\n'.join(summary_lines))
    
    print(f"\\n=== VISUALIZATION COMPLETE ===")
    print(f"Created {visualizations_created} brain visualization(s)")
    print(f"Files saved to: {output_dir}/")
    print(f"Summary: {summary_path}")
    
    # Display key findings
    print(f"\\n=== KEY FINDINGS ===")
    for metric in ['free_energy_change', 'potential_energy_change']:
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            comp_data = roi_averages[roi_averages['comparison'] == comparison]
            if len(comp_data) > 0:
                values = comp_data[metric].values
                metric_name = metric.replace('_', ' ').title()
                comp_name = "During" if 'stimulation' in comparison else "After"
                
                print(f"{metric_name} {comp_name}:")
                print(f"  Mean: {np.mean(values):.4f} ± {np.std(values):.4f}")
                print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
                print(f"  Increases: {np.sum(values > 0)} ROIs")
                print(f"  Decreases: {np.sum(values < 0)} ROIs")

if __name__ == "__main__":
    main()