import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse

def load_and_analyze_results(csv_file):
    """Load and perform comprehensive analysis of Fokker-Planck results."""
    
    print("Loading results...")
    df = pd.read_csv(csv_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Subjects: {df['subject'].nunique()}")
    print(f"ROIs: {df['roi'].nunique()}")
    print(f"Conditions: {list(df['condition'].unique())}")
    print(f"Periods: {list(df['period'].unique())}")
    
    return df

def statistical_analysis(df):
    """Perform statistical analysis comparing active vs sham."""
    
    print("\n=== STATISTICAL ANALYSIS ===")
    
    results = []
    
    for period in ['baseline', 'stimulation', 'recovery']:
        print(f"\n{period.upper()} PERIOD:")
        
        active_data = df[(df['condition'] == 'active') & (df['period'] == period)]
        sham_data = df[(df['condition'] == 'sham') & (df['period'] == period)]
        
        if len(active_data) == 0 or len(sham_data) == 0:
            continue
            
        # Test each coefficient
        for coeff in ['drift_0', 'drift_1', 'diffusion']:
            active_vals = active_data[coeff].values
            sham_vals = sham_data[coeff].values
            
            # t-test
            t_stat, p_val = stats.ttest_ind(active_vals, sham_vals)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(active_vals) - 1) * np.var(active_vals, ddof=1) + 
                                 (len(sham_vals) - 1) * np.var(sham_vals, ddof=1)) / 
                                (len(active_vals) + len(sham_vals) - 2))
            cohens_d = (np.mean(active_vals) - np.mean(sham_vals)) / pooled_std
            
            print(f"  {coeff}:")
            print(f"    Active: {np.mean(active_vals):.4f} ± {np.std(active_vals):.4f}")
            print(f"    Sham:   {np.mean(sham_vals):.4f} ± {np.std(sham_vals):.4f}")
            print(f"    t-test: t={t_stat:.3f}, p={p_val:.3f}")
            print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
            
            results.append({
                'period': period,
                'coefficient': coeff,
                'active_mean': np.mean(active_vals),
                'active_std': np.std(active_vals),
                'sham_mean': np.mean(sham_vals),
                'sham_std': np.std(sham_vals),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'n_active': len(active_vals),
                'n_sham': len(sham_vals)
            })
    
    return pd.DataFrame(results)

def create_visualizations(df, output_prefix='fp_analysis'):
    """Create comprehensive visualizations."""
    
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Coefficient distributions by condition and period
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fokker-Planck Coefficients by Condition and Period', fontsize=16)
    
    coeffs = ['drift_0', 'drift_1', 'diffusion']
    periods = ['baseline', 'stimulation', 'recovery']
    
    for i, coeff in enumerate(coeffs):
        for j, period in enumerate(periods):
            ax = axes[i//2, j] if i < 2 else axes[1, j]
            
            period_data = df[df['period'] == period]
            
            # Box plot
            sns.boxplot(data=period_data, x='condition', y=coeff, ax=ax)
            ax.set_title(f'{coeff} - {period}')
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(coeffs) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(f'../code/{output_prefix}_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Effect sizes across periods
    stats_df = statistical_analysis(df)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Pivot for heatmap
    effect_size_pivot = stats_df.pivot(index='coefficient', columns='period', values='cohens_d')
    
    sns.heatmap(effect_size_pivot, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', ax=ax, cbar_kws={'label': "Cohen's d"})
    ax.set_title('Effect Sizes (Active vs Sham) Across Periods')
    ax.set_xlabel('Period')
    ax.set_ylabel('Coefficient')
    
    plt.tight_layout()
    plt.savefig(f'../code/{output_prefix}_effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. P-values heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    p_value_pivot = stats_df.pivot(index='coefficient', columns='period', values='p_value')
    
    # Create significance annotations
    significance = p_value_pivot.copy()
    significance = significance.applymap(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else '')))
    
    sns.heatmap(p_value_pivot, annot=significance, cmap='Reds_r', 
                fmt='', ax=ax, cbar_kws={'label': 'p-value'})
    ax.set_title('Statistical Significance (Active vs Sham)\n* p<0.05, ** p<0.01, *** p<0.001')
    ax.set_xlabel('Period')
    ax.set_ylabel('Coefficient')
    
    plt.tight_layout()
    plt.savefig(f'../code/{output_prefix}_pvalues.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 4. Time course of coefficients
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, coeff in enumerate(coeffs):
        ax = axes[i]
        
        # Calculate means and SEMs for each period
        period_order = ['baseline', 'stimulation', 'recovery']
        
        for condition in ['active', 'sham']:
            means = []
            sems = []
            
            for period in period_order:
                period_data = df[(df['condition'] == condition) & (df['period'] == period)]
                if len(period_data) > 0:
                    means.append(period_data[coeff].mean())
                    sems.append(period_data[coeff].std() / np.sqrt(len(period_data)))
                else:
                    means.append(np.nan)
                    sems.append(np.nan)
            
            x_pos = np.arange(len(period_order))
            ax.errorbar(x_pos, means, yerr=sems, marker='o', linewidth=2, 
                       markersize=8, label=condition, capsize=5)
        
        ax.set_xlabel('Period')
        ax.set_ylabel(f'{coeff}')
        ax.set_title(f'{coeff} Across Periods')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(period_order)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../code/{output_prefix}_timecourse.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualizations saved with prefix: {output_prefix}")
    
    return stats_df

def subject_level_analysis(df):
    """Analyze individual subject responses."""
    
    print("\n=== SUBJECT-LEVEL ANALYSIS ===")
    
    # Calculate subject-level averages
    subject_stats = df.groupby(['subject', 'condition', 'period']).agg({
        'drift_0': 'mean',
        'drift_1': 'mean', 
        'diffusion': 'mean'
    }).reset_index()
    
    # Calculate differences (active - sham) for each subject and period
    subject_diffs = []
    
    for subject in subject_stats['subject'].unique():
        for period in ['baseline', 'stimulation', 'recovery']:
            active_data = subject_stats[(subject_stats['subject'] == subject) & 
                                      (subject_stats['condition'] == 'active') & 
                                      (subject_stats['period'] == period)]
            sham_data = subject_stats[(subject_stats['subject'] == subject) & 
                                    (subject_stats['condition'] == 'sham') & 
                                    (subject_stats['period'] == period)]
            
            if len(active_data) > 0 and len(sham_data) > 0:
                diff_row = {
                    'subject': subject,
                    'period': period,
                    'drift_0_diff': active_data['drift_0'].iloc[0] - sham_data['drift_0'].iloc[0],
                    'drift_1_diff': active_data['drift_1'].iloc[0] - sham_data['drift_1'].iloc[0],
                    'diffusion_diff': active_data['diffusion'].iloc[0] - sham_data['diffusion'].iloc[0]
                }
                subject_diffs.append(diff_row)
    
    df_diffs = pd.DataFrame(subject_diffs)
    
    # Statistical tests on subject-level differences
    print("Subject-level paired t-tests (Active - Sham):")
    
    for period in ['baseline', 'stimulation', 'recovery']:
        print(f"\n{period.upper()}:")
        period_data = df_diffs[df_diffs['period'] == period]
        
        for coeff in ['drift_0_diff', 'drift_1_diff', 'diffusion_diff']:
            if len(period_data) > 0:
                vals = period_data[coeff].values
                t_stat, p_val = stats.ttest_1samp(vals, 0)  # Test against zero
                
                print(f"  {coeff}: mean={np.mean(vals):.4f}, "
                      f"t={t_stat:.3f}, p={p_val:.3f}")
    
    return df_diffs

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Comprehensive Fokker-Planck analysis')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with results')
    parser.add_argument('--output-prefix', type=str, default='fp_analysis',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Load and analyze data
    df = load_and_analyze_results(args.input)
    
    # Statistical analysis
    stats_df = statistical_analysis(df)
    
    # Create visualizations
    stats_df = create_visualizations(df, args.output_prefix)
    
    # Subject-level analysis
    subject_diffs = subject_level_analysis(df)
    
    # Save statistical results
    stats_df.to_csv(f'../code/{args.output_prefix}_statistics.csv', index=False)
    subject_diffs.to_csv(f'../code/{args.output_prefix}_subject_differences.csv', index=False)
    
    print(f"\nAnalysis complete! Results saved with prefix: {args.output_prefix}")

if __name__ == "__main__":
    main()