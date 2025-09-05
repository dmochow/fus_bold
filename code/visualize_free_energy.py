#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_free_energy_results():
    """Load the free energy analysis results."""
    return pd.read_csv('free_energy_analysis_full.csv')

def create_free_energy_visualizations(df):
    """Create comprehensive visualizations of free energy results."""
    
    print("Creating free energy visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Free energy by period and condition (box plot)
    ax1 = plt.subplot(2, 3, 1)
    sns.boxplot(data=df, x='period', y='free_energy_empirical', hue='condition', ax=ax1)
    ax1.set_title('Free Energy by Period and Condition', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Period', fontsize=12)
    ax1.set_ylabel('Free Energy (empirical)', fontsize=12)
    ax1.legend(title='Condition')
    
    # 2. Free energy changes from baseline
    baseline_data = df[df['period'] == 'baseline'].set_index(['roi_idx', 'condition'])['free_energy_empirical']
    
    changes_data = []
    for period in ['stimulation', 'recovery']:
        period_data = df[df['period'] == period].set_index(['roi_idx', 'condition'])['free_energy_empirical']
        common_idx = baseline_data.index.intersection(period_data.index)
        
        for (roi_idx, condition) in common_idx:
            change = period_data.loc[(roi_idx, condition)] - baseline_data.loc[(roi_idx, condition)]
            changes_data.append({
                'roi_idx': roi_idx,
                'condition': condition,
                'period': period,
                'free_energy_change': change
            })
    
    changes_df = pd.DataFrame(changes_data)
    
    ax2 = plt.subplot(2, 3, 2)
    sns.boxplot(data=changes_df, x='period', y='free_energy_change', hue='condition', ax=ax2)
    ax2.set_title('Free Energy Changes from Baseline', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Period', fontsize=12)
    ax2.set_ylabel('ΔFree Energy (from baseline)', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.legend(title='Condition')
    
    # 3. Paired comparison (Active vs Sham changes)
    active_changes = changes_df[changes_df['condition'] == 'active'].set_index(['roi_idx', 'period'])['free_energy_change']
    sham_changes = changes_df[changes_df['condition'] == 'sham'].set_index(['roi_idx', 'period'])['free_energy_change']
    
    paired_changes = []
    for (roi_idx, period) in active_changes.index:
        if (roi_idx, period) in sham_changes.index:
            diff = active_changes.loc[(roi_idx, period)] - sham_changes.loc[(roi_idx, period)]
            paired_changes.append({
                'roi_idx': roi_idx,
                'period': period,
                'active_minus_sham': diff
            })
    
    paired_df = pd.DataFrame(paired_changes)
    
    ax3 = plt.subplot(2, 3, 3)
    sns.boxplot(data=paired_df, x='period', y='active_minus_sham', ax=ax3)
    ax3.set_title('Active vs Sham Free Energy Changes\\n(Active - Sham)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Period', fontsize=12)
    ax3.set_ylabel('ΔFree Energy (Active - Sham)', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Correlation between components
    ax4 = plt.subplot(2, 3, 4)
    scatter_data = df[df['period'] == 'stimulation']
    scatter = ax4.scatter(scatter_data['potential_energy_empirical'], 
                         scatter_data['entropy_term_empirical'],
                         c=scatter_data['free_energy_empirical'],
                         cmap='viridis', alpha=0.7)
    ax4.set_xlabel('Potential Energy', fontsize=12)
    ax4.set_ylabel('Entropy Term', fontsize=12)
    ax4.set_title('Free Energy Components\\n(Stimulation Period)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Free Energy')
    
    # 5. ROI-specific effects (top 10 most variable)
    roi_variability = df.groupby('roi_idx')['free_energy_empirical'].std().sort_values(ascending=False)
    top_variable_rois = roi_variability.head(10).index
    
    ax5 = plt.subplot(2, 3, 5)
    roi_subset = df[df['roi_idx'].isin(top_variable_rois)]
    
    # Create a pivot table for heatmap
    heatmap_data = roi_subset.pivot_table(
        values='free_energy_empirical', 
        index='roi_idx', 
        columns=['period', 'condition'],
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=heatmap_data.mean().mean(), ax=ax5)
    ax5.set_title('Free Energy Heatmap\\n(Top 10 Most Variable ROIs)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Period and Condition', fontsize=12)
    ax5.set_ylabel('ROI Index', fontsize=12)
    
    # 6. Statistical summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Compute summary statistics
    summary_stats = []
    
    # Overall statistics
    summary_stats.append("FREE ENERGY ANALYSIS SUMMARY")
    summary_stats.append("=" * 40)
    summary_stats.append("")
    summary_stats.append(f"Total ROIs analyzed: {df['roi_idx'].nunique()}")
    summary_stats.append(f"Total measurements: {len(df)}")
    summary_stats.append("")
    
    # Mean values by period
    period_means = df.groupby(['period', 'condition'])['free_energy_empirical'].mean()
    summary_stats.append("Mean Free Energy:")
    for (period, condition), mean_val in period_means.items():
        summary_stats.append(f"  {period} {condition}: {mean_val:.4f}")
    summary_stats.append("")
    
    # Significance of changes
    if len(paired_df) > 0:
        from scipy import stats
        
        summary_stats.append("Statistical Tests (Active vs Sham):")
        for period in ['stimulation', 'recovery']:
            period_data = paired_df[paired_df['period'] == period]['active_minus_sham']
            if len(period_data) > 0:
                t_stat, p_val = stats.ttest_1samp(period_data, 0)
                summary_stats.append(f"  {period}: t={t_stat:.3f}, p={p_val:.4f}")
        summary_stats.append("")
    
    # Effect sizes
    summary_stats.append("Effect Sizes (Cohen's d):")
    for period in ['stimulation', 'recovery']:
        if len(changes_df) > 0:
            active_vals = changes_df[(changes_df['period'] == period) & (changes_df['condition'] == 'active')]['free_energy_change']
            sham_vals = changes_df[(changes_df['period'] == period) & (changes_df['condition'] == 'sham')]['free_energy_change']
            
            if len(active_vals) > 0 and len(sham_vals) > 0:
                pooled_std = np.sqrt(((len(active_vals)-1)*np.var(active_vals) + (len(sham_vals)-1)*np.var(sham_vals)) / (len(active_vals)+len(sham_vals)-2))
                cohens_d = (np.mean(active_vals) - np.mean(sham_vals)) / pooled_std
                summary_stats.append(f"  {period}: d={cohens_d:.3f}")
    
    # Display summary
    summary_text = "\\n".join(summary_stats)
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('free_energy_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Saved: free_energy_analysis.png")
    
    return changes_df, paired_df

def create_individual_roi_plots(df, top_n=5):
    """Create individual plots for top ROIs showing temporal dynamics."""
    
    print(f"Creating individual ROI plots for top {top_n} ROIs...")
    
    # Find ROIs with largest differences between active and sham
    baseline_data = df[df['period'] == 'baseline'].set_index(['roi_idx', 'condition'])['free_energy_empirical']
    stim_data = df[df['period'] == 'stimulation'].set_index(['roi_idx', 'condition'])['free_energy_empirical']
    
    roi_effects = []
    for roi_idx in df['roi_idx'].unique():
        try:
            active_baseline = baseline_data.loc[(roi_idx, 'active')]
            sham_baseline = baseline_data.loc[(roi_idx, 'sham')]
            active_stim = stim_data.loc[(roi_idx, 'active')]
            sham_stim = stim_data.loc[(roi_idx, 'sham')]
            
            # Calculate the difference in changes
            active_change = active_stim - active_baseline
            sham_change = sham_stim - sham_baseline
            effect_size = abs(active_change - sham_change)
            
            roi_effects.append((roi_idx, effect_size))
        except KeyError:
            continue
    
    # Sort by effect size and take top N
    roi_effects.sort(key=lambda x: x[1], reverse=True)
    top_rois = [roi_idx for roi_idx, _ in roi_effects[:top_n]]
    
    # Create individual plots
    fig, axes = plt.subplots(1, top_n, figsize=(4*top_n, 6))
    if top_n == 1:
        axes = [axes]
    
    for i, roi_idx in enumerate(top_rois):
        roi_data = df[df['roi_idx'] == roi_idx]
        roi_label = roi_data.iloc[0]['roi_label']
        
        # Prepare data for plotting
        periods = ['baseline', 'stimulation', 'recovery']
        active_values = []
        sham_values = []
        
        for period in periods:
            period_data = roi_data[roi_data['period'] == period]
            active_val = period_data[period_data['condition'] == 'active']['free_energy_empirical'].iloc[0]
            sham_val = period_data[period_data['condition'] == 'sham']['free_energy_empirical'].iloc[0]
            active_values.append(active_val)
            sham_values.append(sham_val)
        
        # Plot
        ax = axes[i]
        x_pos = np.arange(len(periods))
        
        ax.plot(x_pos, active_values, 'o-', color='red', linewidth=2, markersize=8, label='Active')
        ax.plot(x_pos, sham_values, 'o-', color='blue', linewidth=2, markersize=8, label='Sham')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(periods)
        ax.set_ylabel('Free Energy', fontsize=12)
        ax.set_title(f'ROI {roi_idx}\\n{roi_label[:30]}...', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add effect size annotation
        effect_size = roi_effects[i][1]
        ax.text(0.02, 0.98, f'Effect: {effect_size:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.suptitle('Free Energy Dynamics - Top ROIs by Effect Size', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('free_energy_top_rois.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Saved: free_energy_top_rois.png")

def main():
    """Create all free energy visualizations."""
    
    print("=== CREATING FREE ENERGY VISUALIZATIONS ===")
    
    # Load results
    df = load_free_energy_results()
    
    if len(df) == 0:
        print("No data found. Please run compute_free_energy_full.py first.")
        return
    
    print(f"Loaded {len(df)} measurements from {df['roi_idx'].nunique()} ROIs")
    
    # Create main visualizations
    changes_df, paired_df = create_free_energy_visualizations(df)
    
    # Create individual ROI plots
    create_individual_roi_plots(df, top_n=5)
    
    print("\\n=== VISUALIZATION COMPLETE ===")
    print("Generated files:")
    print("  - free_energy_analysis.png: Comprehensive analysis")
    print("  - free_energy_top_rois.png: Individual ROI dynamics")

if __name__ == "__main__":
    main()