import numpy as np
import pandas as pd
from scipy import stats

def analyze_fokker_planck_results(csv_file):
    """Simple analysis of Fokker-Planck results without heavy plotting."""
    
    print("Loading results...")
    df = pd.read_csv(csv_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Subjects: {df['subject'].nunique()}")
    print(f"ROIs: {df['roi'].nunique()}")
    print(f"Conditions: {list(df['condition'].unique())}")
    print(f"Periods: {list(df['period'].unique())}")
    
    # Statistical analysis
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
            print(f"    Active: {np.mean(active_vals):.4f} ± {np.std(active_vals):.4f} (n={len(active_vals)})")
            print(f"    Sham:   {np.mean(sham_vals):.4f} ± {np.std(sham_vals):.4f} (n={len(sham_vals)})")
            print(f"    Difference: {np.mean(active_vals) - np.mean(sham_vals):.4f}")
            print(f"    t-test: t={t_stat:.3f}, p={p_val:.3f}")
            print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
            
            # Interpretation
            if abs(cohens_d) < 0.2:
                effect_size_interp = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_size_interp = "small"
            elif abs(cohens_d) < 0.8:
                effect_size_interp = "medium"
            else:
                effect_size_interp = "large"
                
            significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            
            print(f"    Interpretation: {effect_size_interp} effect size, {significance}")
            
            results.append({
                'period': period,
                'coefficient': coeff,
                'active_mean': np.mean(active_vals),
                'active_std': np.std(active_vals),
                'sham_mean': np.mean(sham_vals),
                'sham_std': np.std(sham_vals),
                'difference': np.mean(active_vals) - np.mean(sham_vals),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'effect_size_interp': effect_size_interp,
                'significance': significance,
                'n_active': len(active_vals),
                'n_sham': len(sham_vals)
            })
    
    # Save statistical results
    stats_df = pd.DataFrame(results)
    stats_df.to_csv('fokker_planck_statistics.csv', index=False)
    print(f"\nStatistical results saved to 'fokker_planck_statistics.csv'")
    
    # Summary of significant effects
    significant_effects = stats_df[stats_df['p_value'] < 0.05]
    
    if len(significant_effects) > 0:
        print(f"\n=== SIGNIFICANT EFFECTS (p < 0.05) ===")
        for _, row in significant_effects.iterrows():
            print(f"{row['period']} - {row['coefficient']}: "
                  f"d={row['cohens_d']:.3f}, p={row['p_value']:.3f} "
                  f"({row['effect_size_interp']} effect)")
    else:
        print(f"\nNo significant effects found at p < 0.05 level.")
    
    # Period comparisons within conditions
    print(f"\n=== WITHIN-CONDITION COMPARISONS ===")
    
    for condition in ['active', 'sham']:
        print(f"\n{condition.upper()} CONDITION:")
        cond_data = df[df['condition'] == condition]
        
        # Compare stimulation vs baseline
        baseline_data = cond_data[cond_data['period'] == 'baseline']
        stim_data = cond_data[cond_data['period'] == 'stimulation']
        
        if len(baseline_data) > 0 and len(stim_data) > 0:
            print("  Stimulation vs Baseline:")
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                t_stat, p_val = stats.ttest_ind(stim_data[coeff], baseline_data[coeff])
                mean_diff = stim_data[coeff].mean() - baseline_data[coeff].mean()
                print(f"    {coeff}: Δ={mean_diff:.4f}, t={t_stat:.3f}, p={p_val:.3f}")
        
        # Compare recovery vs baseline
        recovery_data = cond_data[cond_data['period'] == 'recovery']
        
        if len(baseline_data) > 0 and len(recovery_data) > 0:
            print("  Recovery vs Baseline:")
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                t_stat, p_val = stats.ttest_ind(recovery_data[coeff], baseline_data[coeff])
                mean_diff = recovery_data[coeff].mean() - baseline_data[coeff].mean()
                print(f"    {coeff}: Δ={mean_diff:.4f}, t={t_stat:.3f}, p={p_val:.3f}")
    
    return stats_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python simple_analysis.py <results_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    stats_df = analyze_fokker_planck_results(csv_file)