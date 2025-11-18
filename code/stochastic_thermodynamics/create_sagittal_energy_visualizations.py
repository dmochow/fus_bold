#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, image
from nilearn.datasets import fetch_atlas_difumo
import warnings
warnings.filterwarnings('ignore')

def load_subject_level_results():
    """Load the detailed component-level free energy results."""
    
    print("Loading detailed component-level free energy results...")
    
    try:
        subject_df = pd.read_csv('free_energy_components_detailed.csv')
        print(f"Loaded {len(subject_df)} records from detailed component analysis")
        return subject_df
    except FileNotFoundError:
        print("Error: free_energy_components_detailed.csv not found")
        print("Please run component analysis first")
        return None

def compute_subject_averages(subject_df):
    """Compute subject averages for each ROI, condition, and period."""
    
    print("Computing subject averages...")
    
    # Group by ROI, condition, and period, then average across subjects
    avg_df = subject_df.groupby(['roi_idx', 'condition', 'period']).agg({
        'free_energy_empirical': 'mean',
        'potential_energy_empirical': 'mean',
        'entropy_term_empirical': 'mean',
        'roi_label': 'first'  # ROI label is the same for all subjects
    }).reset_index()
    
    print(f"Computed averages for {len(avg_df)} ROI-condition-period combinations")
    return avg_df

def compute_energy_changes(avg_df):
    """Compute energy changes from baseline for each ROI and condition."""
    
    print("Computing energy changes from baseline...")
    
    changes_list = []
    
    for roi_idx in avg_df['roi_idx'].unique():
        roi_data = avg_df[avg_df['roi_idx'] == roi_idx]
        roi_label = roi_data['roi_label'].iloc[0]
        
        for condition in ['active', 'sham']:
            condition_data = roi_data[roi_data['condition'] == condition]
            
            # Get baseline values
            baseline_data = condition_data[condition_data['period'] == 'baseline']
            if len(baseline_data) == 0:
                continue
            
            baseline_row = baseline_data.iloc[0]
            
            for period in ['stimulation', 'recovery']:
                period_data = condition_data[condition_data['period'] == period]
                if len(period_data) == 0:
                    continue
                
                period_row = period_data.iloc[0]
                
                # Compute changes
                changes_list.append({
                    'roi_idx': roi_idx,
                    'roi_label': roi_label,
                    'condition': condition,
                    'comparison': f'{period}_minus_baseline',
                    'free_energy_change': period_row['free_energy_empirical'] - baseline_row['free_energy_empirical'],
                    'potential_energy_change': period_row['potential_energy_empirical'] - baseline_row['potential_energy_empirical'],
                    'entropy_change': period_row['entropy_term_empirical'] - baseline_row['entropy_term_empirical']
                })
    
    changes_df = pd.DataFrame(changes_list)
    print(f"Computed changes for {len(changes_df)} ROI-condition-comparison combinations")
    return changes_df

def load_difumo_atlas():
    """Load the DiFuMo atlas."""
    
    print("Loading DiFuMo atlas...")
    
    try:
        # Load DiFuMo 1024 atlas
        atlas = fetch_atlas_difumo(dimension=1024, resolution_mm=2)
        atlas_img = image.load_img(atlas.maps)  # Ensure it's a proper Nifti image
        atlas_labels = atlas.labels
        
        print(f"Loaded DiFuMo atlas with {len(atlas_labels)} regions")
        print(f"Atlas image shape: {atlas_img.shape}")
        print(f"Atlas image affine: {atlas_img.affine}")
        
        return atlas_img, atlas_labels
        
    except Exception as e:
        print(f"Error loading DiFuMo atlas: {e}")
        return None, None

def create_energy_map(changes_df, atlas_img, metric='free_energy_change', 
                     condition='active', comparison='stimulation_minus_baseline'):
    """Create a brain map showing energy changes."""
    
    print(f"Creating {metric} map for {condition} {comparison}...")
    
    # Filter data for the specific condition and comparison
    filtered_data = changes_df[
        (changes_df['condition'] == condition) & 
        (changes_df['comparison'] == comparison)
    ]
    
    if len(filtered_data) == 0:
        print(f"No data found for {condition} {comparison}")
        return None
    
    print(f"Found data for {len(filtered_data)} ROIs")
    
    # Get atlas data
    atlas_data = atlas_img.get_fdata()
    
    # Create energy map
    energy_map = np.zeros_like(atlas_data)
    
    # Map energy values to brain regions
    for _, row in filtered_data.iterrows():
        roi_idx = int(row['roi_idx'])
        energy_value = row[metric]
        
        # Find voxels corresponding to this ROI
        # DiFuMo uses continuous values, so we need to threshold
        roi_mask = atlas_data == (roi_idx + 1)  # DiFuMo is 1-indexed
        
        if np.any(roi_mask):
            energy_map[roi_mask] = energy_value
        else:
            # Try thresholding approach for continuous values
            roi_map = atlas_data[:, :, :, roi_idx] if len(atlas_data.shape) > 3 else atlas_data
            roi_mask = roi_map > 0.001  # Small threshold for DiFuMo
            if np.any(roi_mask):
                energy_map[roi_mask] = energy_value
    
    # Create nibabel image
    energy_img = nib.Nifti1Image(energy_map, atlas_img.affine)
    
    return energy_img

def create_sagittal_visualizations(changes_df, atlas_img):
    """Create sagittal visualizations for all energy metrics and conditions."""
    
    print("Creating sagittal visualizations...")
    
    # Define metrics and their properties
    metrics = {
        'free_energy_change': {
            'title': 'Free Energy Change',
            'cmap': 'RdBu_r',
            'symmetric_cbar': True
        },
        'potential_energy_change': {
            'title': 'Potential Energy Change', 
            'cmap': 'RdBu_r',
            'symmetric_cbar': True
        }
    }
    
    # Define conditions and comparisons
    conditions = ['active']  # Focus on active condition as requested
    comparisons = [
        ('stimulation_minus_baseline', 'During Stimulation'),
        ('recovery_minus_baseline', 'After Stimulation')
    ]
    
    # Create visualization for each metric
    for metric_key, metric_props in metrics.items():
        
        print(f"\\nCreating visualizations for {metric_props['title']}...")
        
        # Create figure with subplots for each comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"{metric_props['title']} - Active vs Baseline", fontsize=16, fontweight='bold')
        
        for i, (comparison, comparison_title) in enumerate(comparisons):
            
            # Create energy map
            energy_img = create_energy_map(
                changes_df, atlas_img, 
                metric=metric_key, 
                condition='active', 
                comparison=comparison
            )
            
            if energy_img is not None:
                
                # Get data range for color scaling
                energy_data = energy_img.get_fdata()
                nonzero_data = energy_data[energy_data != 0]
                
                if len(nonzero_data) > 0:
                    vmin, vmax = np.percentile(nonzero_data, [5, 95])
                    
                    if metric_props['symmetric_cbar']:
                        # Make colorbar symmetric around zero
                        abs_max = max(abs(vmin), abs(vmax))
                        vmin, vmax = -abs_max, abs_max
                    
                    print(f"  {comparison}: Data range [{vmin:.4f}, {vmax:.4f}], {len(nonzero_data)} non-zero voxels")
                    
                    # Create sagittal plot focused on medial regions
                    display = plotting.plot_stat_map(
                        energy_img,
                        bg_img=None,
                        display_mode='x',
                        cut_coords=[-2, 0, 2],  # Medial sagittal slices
                        figure=fig,
                        axes=axes[i],
                        title=comparison_title,
                        colorbar=True,
                        cmap=metric_props['cmap'],
                        vmin=vmin,
                        vmax=vmax,
                        threshold=0.001  # Small threshold to show only non-zero regions
                    )
                    
                else:
                    print(f"  {comparison}: No non-zero data found")
                    axes[i].text(0.5, 0.5, 'No significant\\nchanges detected', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(comparison_title)
            else:
                print(f"  {comparison}: Could not create energy map")
                axes[i].text(0.5, 0.5, 'Could not\\ncreate map', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(comparison_title)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'sagittal_{metric_key}_active_changes.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filename}")
        plt.close()

def create_summary_statistics(changes_df):
    """Create summary statistics for the energy changes."""
    
    print("\\nCreating summary statistics...")
    
    # Filter for active condition
    active_data = changes_df[changes_df['condition'] == 'active']
    
    summary_stats = []
    
    for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
        comp_data = active_data[active_data['comparison'] == comparison]
        
        if len(comp_data) > 0:
            for metric in ['free_energy_change', 'potential_energy_change', 'entropy_change']:
                values = comp_data[metric].values
                
                summary_stats.append({
                    'comparison': comparison,
                    'metric': metric,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_rois': len(values),
                    'n_positive': np.sum(values > 0),
                    'n_negative': np.sum(values < 0)
                })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('energy_changes_summary_statistics.csv', index=False)
    
    print("\\n=== ENERGY CHANGES SUMMARY ===")
    for _, row in summary_df.iterrows():
        comp_name = row['comparison'].replace('_', ' ').title()
        metric_name = row['metric'].replace('_', ' ').title()
        print(f"\\n{comp_name} - {metric_name}:")
        print(f"  Mean: {row['mean']:.4f} ± {row['std']:.4f}")
        print(f"  Range: [{row['min']:.4f}, {row['max']:.4f}]")
        print(f"  ROIs: {row['n_rois']} total, {row['n_positive']} increased, {row['n_negative']} decreased")
    
    return summary_df

def main():
    """Run complete sagittal energy visualization analysis."""
    
    print("=== SAGITTAL ENERGY VISUALIZATIONS ===")
    print("Creating brain maps of free energy and potential energy changes")
    print()
    
    # Load data
    subject_df = load_subject_level_results()
    if subject_df is None:
        return
    
    # Compute subject averages
    avg_df = compute_subject_averages(subject_df)
    
    # Compute energy changes
    changes_df = compute_energy_changes(avg_df)
    
    # Load atlas
    atlas_img, atlas_labels = load_difumo_atlas()
    if atlas_img is None:
        return
    
    # Create visualizations
    create_sagittal_visualizations(changes_df, atlas_img)
    
    # Create summary statistics
    summary_df = create_summary_statistics(changes_df)
    
    print(f"\\n=== VISUALIZATION COMPLETE ===")
    print(f"Created sagittal brain maps showing:")
    print(f"  - Free energy changes during and after active stimulation")
    print(f"  - Potential energy changes during and after active stimulation")
    print(f"  - Subject-averaged values overlaid on medial brain regions")

if __name__ == "__main__":
    main()