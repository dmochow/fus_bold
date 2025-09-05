#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os

def load_time_series_data():
    """Load the DiFuMo time series data."""
    print("Loading time series data...")
    with open('../data/precomputed/difumo_time_series.pkl', 'rb') as f:
        difumo_ts = pickle.load(f)
    
    return difumo_ts

def create_roi_timeseries_plot(difumo_ts, roi_idx, roi_label, roi_key, p_value, effect_size, output_path):
    """Create time series plot for a single ROI showing all subjects and mean."""
    
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
    
    # Main title
    fig.suptitle(f'Time Series: {roi_label} (ROI {roi_idx})\\n'
                f'Drift Coefficient (Baseline): p = {p_value:.6f}, Effect = {effect_size:.4f}', 
                fontsize=16, fontweight='bold')
    
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
        f"Summary Statistics:\\n"
        f"• ROI Index: {roi_idx}\\n"
        f"• Total subjects: 16\\n"
        f"• Total timepoints: 900\\n"
        f"• Baseline: TRs 0-299\\n"
        f"• Stimulation: TRs 300-599\\n"
        f"• Recovery: TRs 600-899\\n"
        f"• Drift₀ effect: {effect_size:.4f}\\n"
        f"• p-value: {p_value:.6f}"
    )
    
    fig.text(0.02, 0.15, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved time series plot: {output_path}")
    
    return True

def main():
    """Create time series visualization for a test ROI."""
    
    # Load time series data
    difumo_ts = load_time_series_data()
    
    # Test with ROI 565 (strongest significant effect)
    roi_idx = 565
    roi_key = 'roi_0565'
    roi_label = 'Middle frontal gyrus posterior LH'
    p_value = 0.000619
    effect_size = 0.0202
    
    # Create output directory
    output_dir = 'timeseries_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    output_path = os.path.join(output_dir, f'timeseries_{roi_key}_{roi_label.replace(" ", "_").replace("/", "-")}.png')
    
    success = create_roi_timeseries_plot(
        difumo_ts, roi_idx, roi_label, roi_key, p_value, effect_size, output_path)
    
    if success:
        print(f"\\n✅ Time series visualization created successfully!")
        print(f"📁 Output: {output_path}")
        print(f"\\nThis shows:")
        print(f"• Individual time series for all 16 subjects (top 4 rows)")
        print(f"• Group mean ± SEM for Active condition (5th row)")
        print(f"• Group mean ± SEM for Sham condition (bottom row)")
        print(f"• Clear period boundaries (baseline/stimulation/recovery)")
    else:
        print("❌ Failed to create time series visualization")

if __name__ == "__main__":
    main()