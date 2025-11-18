import json
import pandas as pd
import numpy as np

def compare_paired_unpaired_results(paired_json, unpaired_json):
    """Compare paired vs unpaired analysis results."""
    
    print("Loading paired analysis results...")
    with open(paired_json, 'r') as f:
        paired_data = json.load(f)
    
    print("Loading unpaired analysis results...")  
    with open(unpaired_json, 'r') as f:
        unpaired_data = json.load(f)
    
    print(f"\n{'='*80}")
    print("COMPARISON: PAIRED vs UNPAIRED STATISTICAL TESTS")
    print(f"{'='*80}")
    
    # Extract results for comparison
    paired_results = []
    unpaired_results = []
    
    # Process paired results
    for roi_key, roi_data in paired_data['roi_results'].items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results'][comparison]['coefficients'][coeff]
                
                if 'p_value_paired' in coeff_data:
                    result = {
                        'roi_index': roi_idx,
                        'roi_label': roi_label,
                        'comparison': comparison,
                        'coefficient': coeff,
                        'p_value': coeff_data['p_value_paired'],
                        'p_value_fdr': coeff_data.get('p_value_paired_fdr', np.nan),
                        'cohens_d': coeff_data['cohens_d_paired'],
                        'paired_difference_mean': coeff_data['paired_difference_mean'],
                        'paired_difference_sem': coeff_data['paired_difference_sem'],
                        'n_subjects': coeff_data['n_paired_subjects'],
                        'active_change_mean': coeff_data['active_change_mean'],
                        'sham_change_mean': coeff_data['sham_change_mean']
                    }
                    paired_results.append(result)
    
    # Process unpaired results (from baseline_corrected_comparisons)
    for roi_key, roi_data in unpaired_data['roi_results'].items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results']['baseline_corrected_comparisons'][comparison]['coefficients'][coeff]
                
                if 'p_value_active_vs_sham' in coeff_data:
                    result = {
                        'roi_index': roi_idx,
                        'roi_label': roi_label,
                        'comparison': comparison,
                        'coefficient': coeff,
                        'p_value': coeff_data['p_value_active_vs_sham'],
                        'p_value_fdr': coeff_data.get('p_value_active_vs_sham_fdr', np.nan),
                        'cohens_d': coeff_data['cohens_d_active_vs_sham'],
                        'change_difference': coeff_data['change_difference'],
                        'active_change_mean': coeff_data['active_change_mean'],
                        'sham_change_mean': coeff_data['sham_change_mean']
                    }
                    unpaired_results.append(result)
    
    df_paired = pd.DataFrame(paired_results)
    df_unpaired = pd.DataFrame(unpaired_results)
    
    print(f"\nData summary:")
    print(f"  Paired results: {len(df_paired)} tests")
    print(f"  Unpaired results: {len(df_unpaired)} tests")
    
    # Overall significance comparison
    paired_sig = len(df_paired[df_paired['p_value'] < 0.05])
    unpaired_sig = len(df_unpaired[df_unpaired['p_value'] < 0.05])
    
    print(f"\nSignificant results (p < 0.05):")
    print(f"  Paired test:   {paired_sig}/{len(df_paired)} ({paired_sig/len(df_paired)*100:.1f}%)")
    print(f"  Unpaired test: {unpaired_sig}/{len(df_unpaired)} ({unpaired_sig/len(df_unpaired)*100:.1f}%)")
    
    # FDR corrected
    paired_sig_fdr = len(df_paired[df_paired['p_value_fdr'] < 0.05])
    unpaired_sig_fdr = len(df_unpaired[df_unpaired['p_value_fdr'] < 0.05])
    
    print(f"\nSignificant results (FDR corrected, p < 0.05):")
    print(f"  Paired test:   {paired_sig_fdr}")
    print(f"  Unpaired test: {unpaired_sig_fdr}")
    
    # Top significant results comparison
    print(f"\n{'='*60}")
    print("TOP 10 MOST SIGNIFICANT RESULTS")
    print(f"{'='*60}")
    
    print(f"\nPAIRED TEST (Within-subject differences):")
    if paired_sig > 0:
        top_paired = df_paired.nsmallest(min(10, paired_sig), 'p_value')
        for i, (_, row) in enumerate(top_paired.iterrows(), 1):
            region_short = row['roi_label'][:40]
            period_short = row['comparison'].replace('_minus_baseline', '')
            print(f"{i:2d}. ROI {row['roi_index']:3d}: {region_short}")
            print(f"     {period_short} {row['coefficient']}: p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
            print(f"     Paired diff: {row['paired_difference_mean']:.4f} ± {row['paired_difference_sem']:.4f}")
            print(f"     Active: {row['active_change_mean']:.4f}, Sham: {row['sham_change_mean']:.4f}")
    else:
        print("  No significant results found")
    
    print(f"\nUNPAIRED TEST (Between-group comparison):")
    if unpaired_sig > 0:
        top_unpaired = df_unpaired.nsmallest(min(10, unpaired_sig), 'p_value')
        for i, (_, row) in enumerate(top_unpaired.iterrows(), 1):
            region_short = row['roi_label'][:40]
            period_short = row['comparison'].replace('_minus_baseline', '')
            print(f"{i:2d}. ROI {row['roi_index']:3d}: {region_short}")
            print(f"     {period_short} {row['coefficient']}: p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
            print(f"     Difference: {row['change_difference']:.4f}")
            print(f"     Active: {row['active_change_mean']:.4f}, Sham: {row['sham_change_mean']:.4f}")
    
    # Create overlap analysis for significant results
    if paired_sig > 0 and unpaired_sig > 0:
        print(f"\n{'='*60}")
        print("OVERLAP ANALYSIS")
        print(f"{'='*60}")
        
        # Create sets of significant ROI-comparison-coefficient combinations
        paired_sig_set = set()
        for _, row in df_paired[df_paired['p_value'] < 0.05].iterrows():
            paired_sig_set.add((row['roi_index'], row['comparison'], row['coefficient']))
        
        unpaired_sig_set = set()
        for _, row in df_unpaired[df_unpaired['p_value'] < 0.05].iterrows():
            unpaired_sig_set.add((row['roi_index'], row['comparison'], row['coefficient']))
        
        overlap = paired_sig_set & unpaired_sig_set
        paired_only = paired_sig_set - unpaired_sig_set
        unpaired_only = unpaired_sig_set - paired_sig_set
        
        print(f"Significant results found in:")
        print(f"  Both methods: {len(overlap)}")
        print(f"  Paired only:  {len(paired_only)}")
        print(f"  Unpaired only: {len(unpaired_only)}")
        
        if len(overlap) > 0:
            print(f"\nOverlapping significant results:")
            for roi_idx, comparison, coeff in list(overlap)[:5]:
                roi_label = df_paired[df_paired['roi_index'] == roi_idx]['roi_label'].iloc[0]
                print(f"  ROI {roi_idx:3d}: {comparison} {coeff} - {roi_label[:40]}")
    
    # Statistical power analysis
    print(f"\n{'='*60}")
    print("STATISTICAL POWER COMPARISON")
    print(f"{'='*60}")
    
    # Merge dataframes for direct comparison
    df_merged = df_paired.merge(
        df_unpaired, 
        on=['roi_index', 'comparison', 'coefficient'], 
        suffixes=('_paired', '_unpaired')
    )
    
    if len(df_merged) > 0:
        print(f"Matched tests for comparison: {len(df_merged)}")
        
        # Compare p-values
        p_correlation = df_merged['p_value_paired'].corr(df_merged['p_value_unpaired'])
        print(f"P-value correlation: {p_correlation:.3f}")
        
        # Cases where paired is more significant
        paired_better = len(df_merged[df_merged['p_value_paired'] < df_merged['p_value_unpaired']])
        unpaired_better = len(df_merged[df_merged['p_value_unpaired'] < df_merged['p_value_paired']])
        
        print(f"Cases where method gives smaller p-value:")
        print(f"  Paired test:   {paired_better} ({paired_better/len(df_merged)*100:.1f}%)")
        print(f"  Unpaired test: {unpaired_better} ({unpaired_better/len(df_merged)*100:.1f}%)")
        
        # Effect size comparison
        effect_correlation = df_merged['cohens_d_paired'].corr(df_merged['cohens_d_unpaired'])
        print(f"Effect size correlation: {effect_correlation:.3f}")
    
    # Summary and recommendations
    print(f"\n{'='*60}")
    print("SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print(f"""
STATISTICAL FINDINGS:

1. POWER DIFFERENCE:
   - Paired test found {paired_sig} significant effects
   - Unpaired test found {unpaired_sig} significant effects
   - Paired test is {'more' if paired_sig > unpaired_sig else 'less'} conservative

2. APPROPRIATE TEST CHOICE:
   - PAIRED TEST is more appropriate for this experimental design
   - Each subject serves as their own control (active vs sham session)
   - Controls for individual differences in baseline brain dynamics
   - Reduces noise from between-subject variability

3. INTERPRETATION:
   - Paired test effects represent true within-subject FUS effects
   - These are the most reliable findings for publication
   - Unpaired effects may include confounding factors

RECOMMENDATION: Use PAIRED TEST results as primary findings.
""")
    
    # Save comparison results
    comparison_summary = {
        'paired_significant': int(paired_sig),
        'unpaired_significant': int(unpaired_sig),
        'paired_total': len(df_paired),
        'unpaired_total': len(df_unpaired)
    }
    
    if paired_sig > 0:
        paired_top = df_paired.nsmallest(min(20, paired_sig), 'p_value')
        paired_effects_csv = "/Users/jacekdmochowski/PROJECTS/fus_bold/code/paired_significant_effects.csv"
        paired_top.to_csv(paired_effects_csv, index=False)
        print(f"\nPaired significant effects saved to: paired_significant_effects.csv")
    
    return df_paired, df_unpaired, comparison_summary

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python compare_paired_unpaired.py <paired_json> <unpaired_json>")
        sys.exit(1)
    
    paired_json = sys.argv[1]
    unpaired_json = sys.argv[2]
    
    df_paired, df_unpaired, summary = compare_paired_unpaired_results(paired_json, unpaired_json)

if __name__ == "__main__":
    main()