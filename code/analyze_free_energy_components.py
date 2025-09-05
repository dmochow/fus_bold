#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_results():
    """Load the subject-level and ROI statistics results."""
    
    subject_df = pd.read_csv('free_energy_subject_level.csv')
    roi_stats_df = pd.read_csv('free_energy_roi_statistics.csv')
    
    return subject_df, roi_stats_df

def analyze_free_energy_components():
    """Analyze potential and entropic components separately."""
    
    print("=== ANALYZING FREE ENERGY COMPONENTS ===")
    print("Tracking potential energy and entropy contributions separately")
    print()
    
    # We need to recompute the analysis to get component-wise statistics
    # Let me modify the subject-level script to save component data
    
    from compute_free_energy_subject_level import (
        load_data, analyze_subject_roi_free_energy, get_roi_label,
        compute_free_energy_changes
    )
    
    # Load data
    difumo_ts, fp_results_df = load_data()
    
    print("Computing component-wise analysis for ALL 1024 ROIs...")
    
    # Analyze ALL ROIs for complete component breakdown
    sample_rois = list(range(1024))  # All 1024 ROIs
    
    component_results = []
    
    for i, roi_idx in enumerate(sample_rois):
        roi_label = get_roi_label(roi_idx)
        if i % 100 == 0:  # Progress update every 100 ROIs
            print(f"Progress: {i+1}/1024 ROIs - Analyzing ROI {roi_idx}: {roi_label[:50]}...")
        
        # Analyze all subjects for this ROI
        roi_subject_results = []
        
        for subject_idx in range(16):
            try:
                subject_results = analyze_subject_roi_free_energy(
                    subject_idx, roi_idx, difumo_ts, fp_results_df)
                roi_subject_results.append(subject_results)
            except Exception as e:
                print(f"  Warning: Error analyzing subject {subject_idx}: {e}")
        
        # Extract component data
        for subject_results in roi_subject_results:
            subject_idx = subject_results['subject_idx']
            
            for period_key, period_data in subject_results['periods'].items():
                condition = period_data['condition']
                period = period_data['period']
                
                component_results.append({
                    'subject_idx': subject_idx,
                    'roi_idx': roi_idx,
                    'roi_label': roi_label,
                    'condition': condition,
                    'period': period,
                    'free_energy_empirical': period_data['free_energy_empirical'],
                    'potential_energy_empirical': period_data['potential_energy_empirical'],
                    'entropy_term_empirical': period_data['entropy_term_empirical'],
                    'free_energy_theoretical': period_data['free_energy_theoretical'],
                    'potential_energy_theoretical': period_data['potential_energy_theoretical'],
                    'entropy_term_theoretical': period_data['entropy_term_theoretical'],
                    'drift_0': period_data['fp_coefficients']['drift_0'],
                    'drift_1': period_data['fp_coefficients']['drift_1'],
                    'diffusion': period_data['fp_coefficients']['diffusion']
                })
    
    component_df = pd.DataFrame(component_results)
    
    # Compute changes from baseline for each component
    print("\\nComputing component-wise changes from baseline...")
    
    component_changes = []
    
    for i, roi_idx in enumerate(sample_rois):
        if i % 100 == 0:  # Progress update every 100 ROIs
            print(f"  Computing changes for ROI {i+1}/1024...")
        for subject_idx in range(16):
            for condition in ['active', 'sham']:
                
                # Get baseline values
                baseline_data = component_df[
                    (component_df['roi_idx'] == roi_idx) & 
                    (component_df['subject_idx'] == subject_idx) & 
                    (component_df['condition'] == condition) & 
                    (component_df['period'] == 'baseline')
                ]
                
                if len(baseline_data) == 0:
                    continue
                
                baseline_row = baseline_data.iloc[0]
                
                for comparison_period in ['stimulation', 'recovery']:
                    
                    # Get comparison period values
                    period_data = component_df[
                        (component_df['roi_idx'] == roi_idx) & 
                        (component_df['subject_idx'] == subject_idx) & 
                        (component_df['condition'] == condition) & 
                        (component_df['period'] == comparison_period)
                    ]
                    
                    if len(period_data) == 0:
                        continue
                    
                    period_row = period_data.iloc[0]
                    
                    # Compute changes
                    component_changes.append({
                        'subject_idx': subject_idx,
                        'roi_idx': roi_idx,
                        'roi_label': baseline_row['roi_label'],
                        'condition': condition,
                        'comparison': f'{comparison_period}_minus_baseline',
                        'free_energy_change': period_row['free_energy_empirical'] - baseline_row['free_energy_empirical'],
                        'potential_energy_change': period_row['potential_energy_empirical'] - baseline_row['potential_energy_empirical'],
                        'entropy_change': period_row['entropy_term_empirical'] - baseline_row['entropy_term_empirical'],
                        'baseline_free_energy': baseline_row['free_energy_empirical'],
                        'baseline_potential': baseline_row['potential_energy_empirical'],
                        'baseline_entropy': baseline_row['entropy_term_empirical'],
                        'period_free_energy': period_row['free_energy_empirical'],
                        'period_potential': period_row['potential_energy_empirical'],
                        'period_entropy': period_row['entropy_term_empirical']
                    })
    
    changes_df = pd.DataFrame(component_changes)
    
    # Perform statistical analysis on components
    print("\\nPerforming component-wise statistical analysis...")
    
    component_stats = []
    
    for i, roi_idx in enumerate(sample_rois):
        if i % 100 == 0:  # Progress update every 100 ROIs
            print(f"  Statistical analysis for ROI {i+1}/1024...")
        roi_label = get_roi_label(roi_idx)
        
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            
            # Get active and sham changes for this ROI and comparison
            active_data = changes_df[
                (changes_df['roi_idx'] == roi_idx) & 
                (changes_df['condition'] == 'active') & 
                (changes_df['comparison'] == comparison)
            ]
            
            sham_data = changes_df[
                (changes_df['roi_idx'] == roi_idx) & 
                (changes_df['condition'] == 'sham') & 
                (changes_df['comparison'] == comparison)
            ]
            
            if len(active_data) > 0 and len(sham_data) > 0 and len(active_data) == len(sham_data):
                
                # Test each component
                for component in ['free_energy_change', 'potential_energy_change', 'entropy_change']:
                    
                    active_vals = active_data[component].values
                    sham_vals = sham_data[component].values
                    
                    # Paired t-test
                    t_stat, p_val = stats.ttest_rel(active_vals, sham_vals)
                    
                    # Effect size
                    cohens_d = (np.mean(active_vals) - np.mean(sham_vals)) / \
                              np.sqrt((np.var(active_vals, ddof=1) + np.var(sham_vals, ddof=1)) / 2)
                    
                    component_stats.append({
                        'roi_idx': roi_idx,
                        'roi_label': roi_label,
                        'comparison': comparison,
                        'component': component,
                        'active_mean': np.mean(active_vals),
                        'active_std': np.std(active_vals, ddof=1),
                        'sham_mean': np.mean(sham_vals),
                        'sham_std': np.std(sham_vals, ddof=1),
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'cohens_d': cohens_d,
                        'n_subjects': len(active_vals)
                    })
    
    component_stats_df = pd.DataFrame(component_stats)
    
    # Save results
    component_df.to_csv('free_energy_components_detailed.csv', index=False)
    changes_df.to_csv('free_energy_component_changes.csv', index=False)
    component_stats_df.to_csv('free_energy_component_statistics.csv', index=False)
    
    print(f"\\nSaved detailed component analysis:")
    print(f"  - free_energy_components_detailed.csv: Raw component values")
    print(f"  - free_energy_component_changes.csv: Component changes from baseline")
    print(f"  - free_energy_component_statistics.csv: Statistical tests for each component")
    
    # Display key findings
    print(f"\\n=== COMPONENT-WISE STATISTICAL FINDINGS ===")
    
    significant_components = component_stats_df[component_stats_df['p_value'] < 0.05]
    
    print(f"Significant component effects (p < 0.05): {len(significant_components)} / {len(component_stats_df)}")
    
    if len(significant_components) > 0:
        print(f"\\nBreakdown by component:")
        component_counts = significant_components.groupby('component').size()
        for component, count in component_counts.items():
            print(f"  {component}: {count} significant effects")
        
        print(f"\\nBreakdown by comparison:")
        comparison_counts = significant_components.groupby('comparison').size()
        for comparison, count in comparison_counts.items():
            print(f"  {comparison}: {count} significant effects")
        
        print(f"\\nTop 5 most significant component effects:")
        top_effects = significant_components.nsmallest(5, 'p_value')[
            ['roi_label', 'comparison', 'component', 'p_value', 'cohens_d']
        ]
        print(top_effects.to_string(index=False))
    
    return component_df, changes_df, component_stats_df

def create_component_visualizations(component_df, changes_df, component_stats_df):
    """Create visualizations for component analysis."""
    
    print("\\nCreating component visualizations...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Component contributions by period and condition
    ax1 = plt.subplot(2, 3, 1)
    
    # Prepare data for stacked bar plot
    period_order = ['baseline', 'stimulation', 'recovery']
    conditions = ['active', 'sham']
    
    mean_components = component_df.groupby(['period', 'condition'])[
        ['potential_energy_empirical', 'entropy_term_empirical']
    ].mean().reset_index()
    
    x_positions = np.arange(len(period_order))
    width = 0.35
    
    for i, condition in enumerate(conditions):
        condition_data = mean_components[mean_components['condition'] == condition]
        condition_data = condition_data.set_index('period').reindex(period_order)
        
        potential_vals = condition_data['potential_energy_empirical'].values
        entropy_vals = condition_data['entropy_term_empirical'].values
        
        x_pos = x_positions + i * width
        
        ax1.bar(x_pos, potential_vals, width, label=f'{condition.title()} - Potential', alpha=0.7)
        ax1.bar(x_pos, entropy_vals, width, bottom=potential_vals, label=f'{condition.title()} - Entropy', alpha=0.7)
    
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Energy Components')
    ax1.set_title('Free Energy Components by Period and Condition')
    ax1.set_xticks(x_positions + width/2)
    ax1.set_xticklabels(period_order)
    ax1.legend()
    
    # 2. Component changes correlation
    ax2 = plt.subplot(2, 3, 2)
    
    scatter = ax2.scatter(changes_df['potential_energy_change'], 
                         changes_df['entropy_change'],
                         c=changes_df['free_energy_change'],
                         cmap='RdBu_r', alpha=0.6)
    ax2.set_xlabel('Potential Energy Change')
    ax2.set_ylabel('Entropy Change')
    ax2.set_title('Potential vs Entropy Changes')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Free Energy Change')
    
    # 3. Component effect sizes
    ax3 = plt.subplot(2, 3, 3)
    
    if len(component_stats_df) > 0:
        significant_comp = component_stats_df[component_stats_df['p_value'] < 0.05]
        
        if len(significant_comp) > 0:
            sns.boxplot(data=significant_comp, x='component', y='cohens_d', ax=ax3)
            ax3.set_title('Effect Sizes by Component\\n(Significant Effects Only)')
            ax3.set_ylabel("Cohen's d")
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Rotate x labels for readability
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No significant\\neffects found', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Effect Sizes by Component')
    
    # 4. Time series of component changes
    ax4 = plt.subplot(2, 3, 4)
    
    # Average component changes across ROIs
    avg_changes = changes_df.groupby(['comparison', 'condition'])[
        ['free_energy_change', 'potential_energy_change', 'entropy_change']
    ].mean().reset_index()
    
    comparisons = ['stimulation_minus_baseline', 'recovery_minus_baseline']
    x_pos = np.arange(len(comparisons))
    width = 0.35
    
    for i, condition in enumerate(['active', 'sham']):
        condition_data = avg_changes[avg_changes['condition'] == condition]
        condition_data = condition_data.set_index('comparison').reindex(comparisons)
        
        fe_changes = condition_data['free_energy_change'].values
        
        ax4.bar(x_pos + i*width, fe_changes, width, 
               label=f'{condition.title()}', alpha=0.7)
    
    ax4.set_xlabel('Comparison')
    ax4.set_ylabel('Mean Free Energy Change')
    ax4.set_title('Free Energy Changes by Condition')
    ax4.set_xticks(x_pos + width/2)
    ax4.set_xticklabels(['Stimulation', 'Recovery'])
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 5. Component heatmap
    ax5 = plt.subplot(2, 3, 5)
    
    if len(component_stats_df) > 0:
        # Create heatmap of effect sizes
        heatmap_data = component_stats_df.pivot_table(
            values='cohens_d', 
            index=['roi_idx', 'roi_label'], 
            columns=['comparison', 'component'],
            aggfunc='mean'
        )
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, ax=ax5, cbar_kws={'label': "Cohen's d"})
            ax5.set_title('Component Effect Sizes Heatmap')
            ax5.set_ylabel('ROI')
        else:
            ax5.text(0.5, 0.5, 'No data for\\nheatmap', 
                    ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary text
    summary_stats = []
    summary_stats.append("FREE ENERGY COMPONENT ANALYSIS")
    summary_stats.append("=" * 40)
    summary_stats.append("")
    summary_stats.append(f"Sample ROIs analyzed: {component_df['roi_idx'].nunique()}")
    summary_stats.append(f"Subjects per ROI: {component_df['subject_idx'].nunique()}")
    summary_stats.append("")
    
    if len(component_stats_df) > 0:
        sig_comp = component_stats_df[component_stats_df['p_value'] < 0.05]
        summary_stats.append(f"Significant effects (p < 0.05):")
        summary_stats.append(f"  Total: {len(sig_comp)} / {len(component_stats_df)}")
        
        if len(sig_comp) > 0:
            comp_breakdown = sig_comp.groupby('component').size()
            for comp, count in comp_breakdown.items():
                comp_name = comp.replace('_change', '').replace('_', ' ').title()
                summary_stats.append(f"  {comp_name}: {count}")
            
            summary_stats.append("")
            summary_stats.append("Mean effect sizes:")
            for comp in ['free_energy_change', 'potential_energy_change', 'entropy_change']:
                comp_data = sig_comp[sig_comp['component'] == comp]['cohens_d']
                if len(comp_data) > 0:
                    comp_name = comp.replace('_change', '').replace('_', ' ').title()
                    summary_stats.append(f"  {comp_name}: {comp_data.mean():.3f}")
    
    summary_text = "\\n".join(summary_stats)
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('free_energy_component_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Saved: free_energy_component_analysis.png")

def main():
    """Run complete component analysis."""
    
    component_df, changes_df, component_stats_df = analyze_free_energy_components()
    create_component_visualizations(component_df, changes_df, component_stats_df)
    
    print(f"\\n=== COMPONENT ANALYSIS COMPLETE ===")
    print(f"Now properly tracking potential energy φ(x) = -(drift_0*x + 0.5*drift_1*x²)/diffusion")
    print(f"and entropy term kT∑p(x)log(p(x)) separately in the analysis")

if __name__ == "__main__":
    main()