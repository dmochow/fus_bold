#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from collections import defaultdict

def load_time_series_data():
    """Load the DiFuMo time series data."""
    print("Loading time series data...")
    with open('../data/precomputed/difumo_time_series.pkl', 'rb') as f:
        difumo_ts = pickle.load(f)
    
    return difumo_ts

def create_roi_timeseries_plot(difumo_ts, roi_info, output_path):
    """Create time series plot for a single ROI showing all subjects and mean."""
    
    roi_idx = roi_info['roi_idx']
    roi_label = roi_info['roi_label']
    roi_key = roi_info['roi_key']
    significant_effects = roi_info['significant_effects']
    
    print(f"Creating time series plot for ROI {roi_idx}: {roi_label}")
    
    # Extract time series for this ROI
    active_ts = np.array([subject_data[:, roi_idx] for subject_data in difumo_ts['active']])  # 16 subjects × 900 TRs
    sham_ts = np.array([subject_data[:, roi_idx] for subject_data in difumo_ts['sham']])    # 16 subjects × 900 TRs
    
    # Create time axis (TRs)
    time_points = np.arange(900)
    
    # Define period boundaries
    baseline_end = 299
    stimulation_end = 599
    
    # Calculate means across subjects
    active_mean = np.mean(active_ts, axis=0)
    sham_mean = np.mean(sham_ts, axis=0)
    
    # Calculate SEM for error bars
    active_sem = np.std(active_ts, axis=0) / np.sqrt(16)
    sham_sem = np.std(sham_ts, axis=0) / np.sqrt(16)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Main title with significant effects summary
    effects_summary = []
    for effect in significant_effects:
        coeff_name = effect['coefficient'].replace('_', ' ').title()
        comp_name = effect['comparison'].replace('_', ' vs ').replace('minus', 'vs').title()
        effects_summary.append(f"{coeff_name} ({comp_name}): p={effect['p_value']:.4f}")
    
    effects_text = " | ".join(effects_summary)
    
    fig.suptitle(f'Time Series: {roi_label} (ROI {roi_idx})\\n'
                f'Significant Effects: {effects_text}', 
                fontsize=14, fontweight='bold')
    
    # Create subplot layout: 4 rows x 5 columns for individual subjects, plus 2 large plots for means
    gs = fig.add_gridspec(6, 5, height_ratios=[1, 1, 1, 1, 2, 2], hspace=0.3, wspace=0.3)
    
    # Plot individual subjects (first 16 subplots)
    colors_active = plt.cm.Reds(np.linspace(0.3, 0.9, 16))
    colors_sham = plt.cm.Blues(np.linspace(0.3, 0.9, 16))
    
    for subj_idx in range(16):
        row = subj_idx // 5
        col = subj_idx % 5
        ax = fig.add_subplot(gs[row, col])
        
        # Plot both conditions for this subject
        ax.plot(time_points, active_ts[subj_idx, :], color=colors_active[subj_idx], 
                linewidth=1, alpha=0.8, label='Active')
        ax.plot(time_points, sham_ts[subj_idx, :], color=colors_sham[subj_idx], 
                linewidth=1, alpha=0.8, label='Sham')
        
        # Add period boundaries
        ax.axvline(x=baseline_end, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=stimulation_end, color='gray', linestyle='--', alpha=0.5)
        
        # Format subplot
        ax.set_title(f'Subject {subj_idx+1}', fontsize=10)
        ax.set_xlim(0, 899)
        ax.tick_params(labelsize=8)
        
        # Add legend only to first subplot
        if subj_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Remove x labels except bottom row
        if row < 3:
            ax.set_xticklabels([])
    
    # Plot group means (bottom two large subplots)
    # Active condition mean
    ax_active = fig.add_subplot(gs[4, :])
    
    # Plot individual subjects as thin lines
    for subj_idx in range(16):
        ax_active.plot(time_points, active_ts[subj_idx, :], color='red', alpha=0.2, linewidth=0.5)
    
    # Plot mean with error bars
    ax_active.plot(time_points, active_mean, color='darkred', linewidth=3, label='Group Mean')
    ax_active.fill_between(time_points, active_mean - active_sem, active_mean + active_sem, 
                          color='red', alpha=0.3, label='SEM')
    
    # Add period boundaries and labels
    ax_active.axvline(x=baseline_end, color='black', linestyle='--', alpha=0.7)
    ax_active.axvline(x=stimulation_end, color='black', linestyle='--', alpha=0.7)
    
    # Add period labels
    ax_active.text(150, ax_active.get_ylim()[1]*0.9, 'Baseline', ha='center', fontsize=12, 
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax_active.text(450, ax_active.get_ylim()[1]*0.9, 'Stimulation', ha='center', fontsize=12,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    ax_active.text(750, ax_active.get_ylim()[1]*0.9, 'Recovery', ha='center', fontsize=12,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax_active.set_title('Active FUS Condition', fontsize=14, fontweight='bold')
    ax_active.set_ylabel('BOLD Signal', fontsize=12)
    ax_active.set_xlim(0, 899)
    ax_active.legend(loc='upper right')
    ax_active.grid(True, alpha=0.3)
    
    # Sham condition mean
    ax_sham = fig.add_subplot(gs[5, :])
    
    # Plot individual subjects as thin lines
    for subj_idx in range(16):
        ax_sham.plot(time_points, sham_ts[subj_idx, :], color='blue', alpha=0.2, linewidth=0.5)
    
    # Plot mean with error bars
    ax_sham.plot(time_points, sham_mean, color='darkblue', linewidth=3, label='Group Mean')
    ax_sham.fill_between(time_points, sham_mean - sham_sem, sham_mean + sham_sem, 
                        color='blue', alpha=0.3, label='SEM')
    
    # Add period boundaries and labels
    ax_sham.axvline(x=baseline_end, color='black', linestyle='--', alpha=0.7)
    ax_sham.axvline(x=stimulation_end, color='black', linestyle='--', alpha=0.7)
    
    # Add period labels
    ax_sham.text(150, ax_sham.get_ylim()[1]*0.9, 'Baseline', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax_sham.text(450, ax_sham.get_ylim()[1]*0.9, 'Stimulation', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    ax_sham.text(750, ax_sham.get_ylim()[1]*0.9, 'Recovery', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax_sham.set_title('Sham FUS Condition', fontsize=14, fontweight='bold')
    ax_sham.set_xlabel('Time (TRs)', fontsize=12)
    ax_sham.set_ylabel('BOLD Signal', fontsize=12)
    ax_sham.set_xlim(0, 899)
    ax_sham.legend(loc='upper right')
    ax_sham.grid(True, alpha=0.3)
    
    # Add summary statistics text box
    stats_text = (
        f"ROI Information:\\n"
        f"• Index: {roi_idx}\\n"
        f"• Total subjects: 16\\n"
        f"• Timepoints: 900 TRs\\n"
        f"• Baseline: 0-299\\n"
        f"• Stimulation: 300-599\\n"
        f"• Recovery: 600-899\\n"
        f"\\nSignificant Effects:\\n"
    )
    
    for effect in significant_effects:
        stats_text += f"• {effect['coefficient']}: p={effect['p_value']:.4f}\\n"
    
    fig.text(0.02, 0.15, stats_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {os.path.basename(output_path)}")
    
    return True

def find_all_significant_rois(results_data):
    """Find all ROIs with significant effects and organize their information."""
    
    significant_rois = defaultdict(lambda: {
        'roi_idx': None,
        'roi_key': None, 
        'roi_label': None,
        'significant_effects': []
    })
    
    coefficients = ['drift_0', 'drift_1', 'diffusion']
    comparisons = ['stimulation_minus_baseline', 'recovery_minus_baseline']
    
    for roi_key, roi_data in results_data['roi_results'].items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for comparison in comparisons:
            for coefficient in coefficients:
                coeff_data = roi_data['analysis_results'][comparison]['coefficients'][coefficient]
                p_val = coeff_data.get('p_value_paired', 1.0)
                
                if p_val < 0.05:
                    if roi_idx not in significant_rois:
                        significant_rois[roi_idx]['roi_idx'] = roi_idx
                        significant_rois[roi_idx]['roi_key'] = roi_key
                        significant_rois[roi_idx]['roi_label'] = roi_label
                    
                    significant_rois[roi_idx]['significant_effects'].append({
                        'coefficient': coefficient,
                        'comparison': comparison,
                        'p_value': p_val,
                        'effect_size': coeff_data.get('paired_difference_mean', 0)
                    })
    
    return dict(significant_rois)

def main():
    """Create time series visualizations for all significant ROIs."""
    
    print("=== CREATING TIME SERIES VISUALIZATIONS FOR ALL SIGNIFICANT ROIS ===")
    
    # Load paired results to find significant ROIs
    print("Loading statistical results...")
    with open('paired_full_results.json', 'r') as f:
        results_data = json.load(f)
    
    # Find all significant ROIs
    print("Finding all significant ROIs...")
    significant_rois = find_all_significant_rois(results_data)
    
    print(f"Found {len(significant_rois)} ROIs with significant effects")
    
    # Load time series data
    difumo_ts = load_time_series_data()
    
    # Create output directory
    output_dir = 'timeseries_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations for all significant ROIs
    successful_plots = 0
    failed_plots = 0
    
    # Sort ROIs by index for organized processing
    sorted_rois = sorted(significant_rois.items(), key=lambda x: x[0])
    
    for roi_idx, roi_info in sorted_rois:
        try:
            # Create safe filename
            safe_label = roi_info['roi_label'].replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '')
            output_path = os.path.join(output_dir, f'timeseries_roi_{roi_idx:04d}_{safe_label}.png')
            
            # Create the plot
            success = create_roi_timeseries_plot(difumo_ts, roi_info, output_path)
            
            if success:
                successful_plots += 1
            else:
                failed_plots += 1
                
        except Exception as e:
            print(f"  ERROR processing ROI {roi_idx}: {e}")
            failed_plots += 1
    
    # Create summary report
    summary_report = []
    summary_report.append("FUS-BOLD FOKKER-PLANCK - TIME SERIES VISUALIZATIONS SUMMARY")
    summary_report.append("=" * 70)
    summary_report.append("")
    summary_report.append(f"Total significant ROIs processed: {len(significant_rois)}")
    summary_report.append(f"Successful visualizations: {successful_plots}")
    summary_report.append(f"Failed visualizations: {failed_plots}")
    summary_report.append("")
    summary_report.append("ROI Details:")
    summary_report.append("-" * 50)
    
    for roi_idx, roi_info in sorted_rois:
        summary_report.append(f"ROI {roi_idx:04d}: {roi_info['roi_label']}")
        summary_report.append(f"  Significant effects: {len(roi_info['significant_effects'])}")
        for effect in roi_info['significant_effects']:
            coeff_name = effect['coefficient'].replace('_', ' ')
            comp_name = effect['comparison'].replace('_', ' vs ')
            summary_report.append(f"    {coeff_name} ({comp_name}): p={effect['p_value']:.4f}")
        summary_report.append("")
    
    # Save summary
    summary_path = os.path.join(output_dir, "timeseries_summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\\n'.join(summary_report))
    
    print(f"\\n=== TIME SERIES VISUALIZATION COMPLETE ===")
    print(f"📊 Successfully created {successful_plots} time series visualizations")
    print(f"❌ Failed to create {failed_plots} visualizations")
    print(f"📁 Files saved to: {output_dir}")
    print(f"📋 Summary report: {summary_path}")
    
    if successful_plots > 0:
        print(f"\\n✅ Each visualization includes:")
        print(f"   • Individual time series for all 16 subjects")
        print(f"   • Group mean ± SEM for Active and Sham conditions")
        print(f"   • Clear experimental period boundaries")
        print(f"   • Statistical significance information")

if __name__ == "__main__":
    main()