#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_energy_changes():
    """Load and analyze energy changes from the component data."""
    
    print("Loading component changes data...")
    
    try:
        changes_df = pd.read_csv('free_energy_component_changes.csv')
        print(f"Loaded {len(changes_df)} change records")
        
        # Focus on active condition as requested
        active_changes = changes_df[changes_df['condition'] == 'active']
        print(f"Found {len(active_changes)} active condition records")
        
        return active_changes
        
    except FileNotFoundError:
        print("Error: Component changes file not found")
        return None

def create_energy_summary_plots(active_changes):
    """Create summary plots of energy changes."""
    
    print("Creating energy summary visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Free Energy and Potential Energy Changes - Active Stimulation', 
                 fontsize=16, fontweight='bold')
    
    # 1. Free Energy Changes - Stimulation vs Baseline
    ax1 = axes[0, 0]
    stim_data = active_changes[active_changes['comparison'] == 'stimulation_minus_baseline']
    
    if len(stim_data) > 0:
        fe_changes = stim_data['free_energy_change'].values
        ax1.hist(fe_changes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(fe_changes), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(fe_changes):.4f}')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Free Energy Change')
        ax1.set_ylabel('Number of ROIs')
        ax1.set_title('During Stimulation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Free Energy Changes - Recovery vs Baseline  
    ax2 = axes[0, 1]
    rec_data = active_changes[active_changes['comparison'] == 'recovery_minus_baseline']
    
    if len(rec_data) > 0:
        fe_changes = rec_data['free_energy_change'].values
        ax2.hist(fe_changes, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(np.mean(fe_changes), color='red', linestyle='--',
                   label=f'Mean: {np.mean(fe_changes):.4f}')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Free Energy Change')
        ax2.set_ylabel('Number of ROIs')
        ax2.set_title('After Stimulation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Comparison of Free Energy Changes
    ax3 = axes[0, 2]
    
    if len(stim_data) > 0 and len(rec_data) > 0:
        # Align ROIs for comparison
        stim_roi_data = stim_data.set_index('roi_idx')['free_energy_change']
        rec_roi_data = rec_data.set_index('roi_idx')['free_energy_change']
        
        common_rois = stim_roi_data.index.intersection(rec_roi_data.index)
        
        if len(common_rois) > 0:
            stim_aligned = stim_roi_data.loc[common_rois]
            rec_aligned = rec_roi_data.loc[common_rois]
            
            ax3.scatter(stim_aligned, rec_aligned, alpha=0.6, s=30)
            ax3.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5)  # Diagonal line
            ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
            ax3.set_xlabel('During Stimulation')
            ax3.set_ylabel('After Stimulation')
            ax3.set_title('Free Energy Changes\\nComparison')
            ax3.grid(True, alpha=0.3)
            
            # Add correlation
            corr = np.corrcoef(stim_aligned, rec_aligned)[0, 1]
            ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Potential Energy Changes - Stimulation vs Baseline
    ax4 = axes[1, 0]
    
    if len(stim_data) > 0:
        pe_changes = stim_data['potential_energy_change'].values
        ax4.hist(pe_changes, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(np.mean(pe_changes), color='red', linestyle='--',
                   label=f'Mean: {np.mean(pe_changes):.4f}')
        ax4.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Potential Energy Change')
        ax4.set_ylabel('Number of ROIs')
        ax4.set_title('Potential Energy - During Stimulation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Potential Energy Changes - Recovery vs Baseline
    ax5 = axes[1, 1]
    
    if len(rec_data) > 0:
        pe_changes = rec_data['potential_energy_change'].values
        ax5.hist(pe_changes, bins=50, alpha=0.7, color='gold', edgecolor='black')
        ax5.axvline(np.mean(pe_changes), color='red', linestyle='--',
                   label=f'Mean: {np.mean(pe_changes):.4f}')
        ax5.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax5.set_xlabel('Potential Energy Change')
        ax5.set_ylabel('Number of ROIs')
        ax5.set_title('Potential Energy - After Stimulation')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Component Comparison
    ax6 = axes[1, 2]
    
    # Create box plots comparing components across periods
    plot_data = []
    
    for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
        comp_data = active_changes[active_changes['comparison'] == comparison]
        period_name = 'During' if 'stimulation' in comparison else 'After'
        
        for component in ['free_energy_change', 'potential_energy_change', 'entropy_change']:
            values = comp_data[component].values
            comp_name = component.replace('_change', '').replace('_', ' ').title()
            
            for val in values:
                plot_data.append({
                    'Period': period_name,
                    'Component': comp_name,
                    'Value': val
                })
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        
        # Box plot for free energy and potential energy only
        fe_pe_data = plot_df[plot_df['Component'].isin(['Free Energy', 'Potential Energy'])]
        
        if len(fe_pe_data) > 0:
            sns.boxplot(data=fe_pe_data, x='Period', y='Value', hue='Component', ax=ax6)
            ax6.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax6.set_title('Energy Components\\nComparison')
            ax6.grid(True, alpha=0.3)
            ax6.legend(title='Component')
    
    plt.tight_layout()
    plt.savefig('energy_changes_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: energy_changes_summary.png")
    plt.close()

def create_statistics_summary(active_changes):
    """Create detailed statistics summary."""
    
    print("\\nCreating statistics summary...")
    
    summary_stats = []
    
    for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
        comp_data = active_changes[active_changes['comparison'] == comparison]
        period_name = 'During Stimulation' if 'stimulation' in comparison else 'After Stimulation'
        
        if len(comp_data) > 0:
            for metric in ['free_energy_change', 'potential_energy_change', 'entropy_change']:
                values = comp_data[metric].values
                metric_name = metric.replace('_change', '').replace('_', ' ').title()
                
                summary_stats.append({
                    'Period': period_name,
                    'Metric': metric_name,
                    'Mean': np.mean(values),
                    'Std': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'N_ROIs': len(values),
                    'N_Positive': np.sum(values > 0),
                    'N_Negative': np.sum(values < 0),
                    'Percent_Positive': (np.sum(values > 0) / len(values)) * 100,
                    'Mean_Magnitude': np.mean(np.abs(values))
                })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('energy_changes_subject_averaged_summary.csv', index=False)
    
    print("\\n=== ENERGY CHANGES SUMMARY (Subject-Averaged) ===")
    
    for period in ['During Stimulation', 'After Stimulation']:
        period_data = summary_df[summary_df['Period'] == period]
        
        print(f"\\n{period}:")
        print("-" * 50)
        
        for _, row in period_data.iterrows():
            print(f"\\n{row['Metric']}:")
            print(f"  Mean change: {row['Mean']:.4f} ± {row['Std']:.4f}")
            print(f"  Range: [{row['Min']:.4f}, {row['Max']:.4f}]")
            print(f"  ROIs: {row['N_ROIs']} total")
            print(f"  Direction: {row['N_Positive']} increased ({row['Percent_Positive']:.1f}%), {row['N_Negative']} decreased")
            print(f"  Mean magnitude: {row['Mean_Magnitude']:.4f}")
    
    return summary_df

def main():
    """Run energy summary analysis."""
    
    print("=== ENERGY CHANGES SUMMARY ANALYSIS ===")
    print("Subject-averaged energy changes during and after active FUS stimulation")
    print()
    
    # Load data
    active_changes = load_and_analyze_energy_changes()
    if active_changes is None:
        return
    
    # Create visualizations
    create_energy_summary_plots(active_changes)
    
    # Create statistics
    summary_df = create_statistics_summary(active_changes)
    
    print(f"\\n=== ANALYSIS COMPLETE ===")
    print(f"Created energy changes summary showing:")
    print(f"  - Subject-averaged free energy changes")
    print(f"  - Subject-averaged potential energy changes") 
    print(f"  - Statistical summaries for during and after stimulation")
    print(f"  - Comparison plots between stimulation periods")

if __name__ == "__main__":
    main()