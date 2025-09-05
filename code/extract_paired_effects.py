import json
import pandas as pd
import numpy as np

def extract_paired_significant_effects(json_file, alpha=0.05):
    """Extract significant effects from paired analysis."""
    
    print(f"Loading paired analysis results from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    analysis_info = data['analysis_info']
    roi_results = data['roi_results']
    
    print(f"Analysis contains {analysis_info['n_rois_processed']} ROIs")
    print(f"Extracting paired effects with p-value < {alpha}")
    
    significant_effects = []
    
    for roi_key, roi_data in roi_results.items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results'][comparison]['coefficients'][coeff]
                
                if 'p_value_paired' in coeff_data:
                    p_val = coeff_data['p_value_paired']
                    if p_val < alpha:
                        effect = {
                            'roi_index': roi_idx,
                            'roi_label': roi_label,
                            'comparison': comparison,
                            'coefficient': coeff,
                            'p_value_paired': p_val,
                            'p_value_paired_fdr': coeff_data.get('p_value_paired_fdr', np.nan),
                            'p_value_paired_bonferroni': coeff_data.get('p_value_paired_bonferroni', np.nan),
                            'cohens_d_paired': coeff_data.get('cohens_d_paired', np.nan),
                            't_statistic_paired': coeff_data.get('t_statistic_paired', np.nan),
                            'degrees_of_freedom': coeff_data.get('degrees_of_freedom_paired', np.nan),
                            'n_paired_subjects': coeff_data.get('n_paired_subjects', np.nan),
                            
                            # Active condition changes
                            'active_change_mean': coeff_data.get('active_change_mean', np.nan),
                            'active_change_std': coeff_data.get('active_change_std', np.nan),
                            'active_change_sem': coeff_data.get('active_change_sem', np.nan),
                            
                            # Sham condition changes
                            'sham_change_mean': coeff_data.get('sham_change_mean', np.nan),
                            'sham_change_std': coeff_data.get('sham_change_std', np.nan),
                            'sham_change_sem': coeff_data.get('sham_change_sem', np.nan),
                            
                            # Paired differences
                            'paired_difference_mean': coeff_data.get('paired_difference_mean', np.nan),
                            'paired_difference_std': coeff_data.get('paired_difference_std', np.nan),
                            'paired_difference_sem': coeff_data.get('paired_difference_sem', np.nan),
                            
                            # One-sample tests
                            'p_value_active_vs_zero': coeff_data.get('p_value_active_vs_zero', np.nan),
                            'p_value_sham_vs_zero': coeff_data.get('p_value_sham_vs_zero', np.nan),
                            
                            # Unpaired comparison for reference
                            'p_value_unpaired': coeff_data.get('p_value_unpaired', np.nan),
                            'cohens_d_unpaired': coeff_data.get('cohens_d_unpaired', np.nan)
                        }
                        significant_effects.append(effect)
    
    # Convert to DataFrame
    df_significant = pd.DataFrame(significant_effects)
    
    if len(df_significant) == 0:
        print(f"No significant paired effects found at α = {alpha}")
        return df_significant
    
    # Sort by p-value
    df_significant = df_significant.sort_values('p_value_paired').reset_index(drop=True)
    
    # Add effect size categories
    df_significant['effect_size_category'] = df_significant['cohens_d_paired'].apply(
        lambda d: 'large' if abs(d) >= 0.8 else ('medium' if abs(d) >= 0.5 else ('small' if abs(d) >= 0.2 else 'negligible'))
    )
    
    # Add change directions
    df_significant['active_change_direction'] = df_significant['active_change_mean'].apply(
        lambda x: 'increase' if x > 0 else ('decrease' if x < 0 else 'no_change') if not pd.isna(x) else 'N/A'
    )
    df_significant['sham_change_direction'] = df_significant['sham_change_mean'].apply(
        lambda x: 'increase' if x > 0 else ('decrease' if x < 0 else 'no_change') if not pd.isna(x) else 'N/A'
    )
    
    # Add consistency indicator
    df_significant['change_pattern'] = df_significant.apply(
        lambda row: f"A{row['active_change_direction'][0].upper()}S{row['sham_change_direction'][0].upper()}", axis=1
    )
    
    # Summary statistics
    print(f"\n=== PAIRED SIGNIFICANT EFFECTS SUMMARY ===")
    print(f"Total significant paired effects: {len(df_significant)}")
    
    # By comparison
    comparison_counts = df_significant['comparison'].value_counts()
    print(f"\nBy comparison:")
    for comparison, count in comparison_counts.items():
        print(f"  {comparison}: {count}")
    
    # By coefficient
    coeff_counts = df_significant['coefficient'].value_counts()
    print(f"\nBy coefficient:")
    for coeff, count in coeff_counts.items():
        print(f"  {coeff}: {count}")
    
    # Effect sizes
    effect_size_counts = df_significant['effect_size_category'].value_counts()
    print(f"\nBy effect size:")
    for effect_size, count in effect_size_counts.items():
        print(f"  {effect_size}: {count}")
    
    # Change patterns
    pattern_counts = df_significant['change_pattern'].value_counts()
    print(f"\nChange patterns (Active/Sham):")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count}")
    
    # Top ROIs with most effects
    roi_counts = df_significant['roi_index'].value_counts().head(10)
    print(f"\nTop 10 ROIs with most significant paired effects:")
    for roi_idx, count in roi_counts.items():
        roi_label = df_significant[df_significant['roi_index'] == roi_idx]['roi_label'].iloc[0]
        print(f"  ROI {roi_idx:3d}: {count} effects - {roi_label[:50]}...")
    
    # FDR corrected results
    fdr_significant = len(df_significant[df_significant['p_value_paired_fdr'] < alpha])
    bonf_significant = len(df_significant[df_significant['p_value_paired_bonferroni'] < alpha])
    
    print(f"\nMultiple comparison corrections:")
    print(f"  FDR corrected (p < {alpha}): {fdr_significant}")
    print(f"  Bonferroni corrected (p < {alpha}): {bonf_significant}")
    
    # Save to CSV
    output_csv = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/paired_significant_effects_full.csv"
    df_significant.to_csv(output_csv, index=False)
    print(f"\nPaired significant effects saved to: paired_significant_effects_full.csv")
    
    # Create summary of top 20
    top_20 = df_significant.head(20)[
        ['roi_index', 'roi_label', 'comparison', 'coefficient', 
         'p_value_paired', 'cohens_d_paired', 'paired_difference_mean', 
         'paired_difference_sem', 'active_change_mean', 'sham_change_mean', 
         'change_pattern', 'n_paired_subjects']
    ]
    
    top_20_csv = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/top_20_paired_effects.csv"
    top_20.to_csv(top_20_csv, index=False)
    print(f"Top 20 paired effects saved to: top_20_paired_effects.csv")
    
    return df_significant

def create_paired_summary_report(df_significant):
    """Create a comprehensive summary report."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE PAIRED ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    if len(df_significant) == 0:
        print("No significant effects found.")
        return
    
    print(f"\nTOP 20 MOST SIGNIFICANT PAIRED EFFECTS:")
    print(f"{'Rank':<4} {'ROI':<4} {'Region':<35} {'Period':<12} {'Coeff':<8} {'p-value':<10} {'Effect':<6} {'Paired Δ':<10}")
    print("-" * 95)
    
    for i, (_, row) in enumerate(df_significant.head(20).iterrows(), 1):
        region_short = row['roi_label'][:34]
        period_short = row['comparison'].replace('_minus_baseline', '').replace('_', ' ')
        
        print(f"{i:<4} {row['roi_index']:<4} {region_short:<35} {period_short:<12} {row['coefficient']:<8} "
              f"{row['p_value_paired']:<10.4f} {row['change_pattern']:<6} {row['paired_difference_mean']:<10.4f}")
    
    # Network analysis
    print(f"\n{'='*60}")
    print("BRAIN NETWORK ANALYSIS")
    print(f"{'='*60}")
    
    networks = {
        'Cerebellum': ['cerebellum'],
        'Visual': ['visual', 'occipital', 'calcarine', 'cuneus'],
        'Sensorimotor': ['precentral', 'postcentral', 'motor', 'central sulcus'],
        'Frontal': ['frontal', 'prefrontal'],
        'Parietal': ['parietal'],
        'Temporal': ['temporal'],
        'Cingulate': ['cingulate'],
        'Insula': ['insula'],
        'Subcortical': ['thalamus', 'caudate', 'putamen', 'hippocampus'],
        'White Matter': ['fasciculus', 'radiation', 'tract', 'capsule', 'corpus callosum']
    }
    
    for network, keywords in networks.items():
        network_effects = df_significant[
            df_significant['roi_label'].str.contains('|'.join(keywords), case=False, na=False)
        ]
        if len(network_effects) > 0:
            stimulation_effects = len(network_effects[network_effects['comparison'] == 'stimulation_minus_baseline'])
            recovery_effects = len(network_effects[network_effects['comparison'] == 'recovery_minus_baseline'])
            print(f"{network:<15}: {len(network_effects):3d} effects ({stimulation_effects} stim, {recovery_effects} rec)")
    
    # Direction consistency analysis
    print(f"\n{'='*60}")
    print("EFFECT DIRECTION ANALYSIS")
    print(f"{'='*60}")
    
    # Most common patterns
    pattern_analysis = df_significant['change_pattern'].value_counts()
    print(f"Most common change patterns:")
    for pattern, count in pattern_analysis.head(5).items():
        pattern_desc = {
            'AISD': 'Active increases, Sham decreases',
            'ADSI': 'Active decreases, Sham increases', 
            'AISI': 'Active increases, Sham increases',
            'ADSD': 'Active decreases, Sham decreases'
        }
        desc = pattern_desc.get(pattern, pattern)
        print(f"  {pattern} ({desc}): {count} effects")
    
    # Statistical power summary
    print(f"\n{'='*60}")
    print("STATISTICAL POWER SUMMARY")
    print(f"{'='*60}")
    
    print(f"Mean number of paired subjects: {df_significant['n_paired_subjects'].mean():.1f}")
    print(f"Range of subjects: {df_significant['n_paired_subjects'].min()}-{df_significant['n_paired_subjects'].max()}")
    
    # Effect size distribution
    print(f"\nPaired effect sizes (Cohen's d):")
    print(f"  Mean |d|: {abs(df_significant['cohens_d_paired']).mean():.3f}")
    print(f"  Median |d|: {abs(df_significant['cohens_d_paired']).median():.3f}")
    print(f"  Large effects (|d| ≥ 0.8): {len(df_significant[abs(df_significant['cohens_d_paired']) >= 0.8])}")

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python extract_paired_effects.py <paired_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    df_significant = extract_paired_significant_effects(json_file)
    
    if len(df_significant) > 0:
        create_paired_summary_report(df_significant)

if __name__ == "__main__":
    main()