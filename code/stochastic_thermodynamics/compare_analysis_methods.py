import json
import pandas as pd
import numpy as np

def compare_analysis_methods(json_file):
    """Compare standard vs baseline-corrected analysis results."""
    
    print(f"Loading baseline-corrected analysis results from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    analysis_info = data['analysis_info']
    roi_results = data['roi_results']
    
    print(f"Analysis contains {analysis_info['n_rois_processed']} ROIs")
    
    # Extract standard comparison results
    standard_results = []
    for roi_key, roi_data in roi_results.items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for period in ['baseline', 'stimulation', 'recovery']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results']['standard_comparisons'][period]['coefficients'][coeff]
                
                if 'p_value' in coeff_data:
                    result = {
                        'analysis_type': 'standard',
                        'roi_index': roi_idx,
                        'roi_label': roi_label,
                        'comparison': period,
                        'coefficient': coeff,
                        'p_value': coeff_data['p_value'],
                        'p_value_fdr': coeff_data.get('p_value_fdr', np.nan),
                        'cohens_d': coeff_data.get('cohens_d', np.nan),
                        'mean_difference': coeff_data.get('mean_difference', np.nan)
                    }
                    standard_results.append(result)
    
    # Extract baseline-corrected comparison results
    baseline_corrected_results = []
    for roi_key, roi_data in roi_results.items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results']['baseline_corrected_comparisons'][comparison]['coefficients'][coeff]
                
                if 'p_value_active_vs_sham' in coeff_data:
                    # Active vs Sham comparison (primary interest)
                    result = {
                        'analysis_type': 'baseline_corrected',
                        'roi_index': roi_idx,
                        'roi_label': roi_label,
                        'comparison': comparison,
                        'coefficient': coeff,
                        'p_value': coeff_data['p_value_active_vs_sham'],
                        'p_value_fdr': coeff_data.get('p_value_active_vs_sham_fdr', np.nan),
                        'cohens_d': coeff_data.get('cohens_d_active_vs_sham', np.nan),
                        'change_difference': coeff_data.get('change_difference', np.nan),
                        'active_change_mean': coeff_data.get('active_change_mean', np.nan),
                        'sham_change_mean': coeff_data.get('sham_change_mean', np.nan),
                        'p_value_active_vs_zero': coeff_data.get('p_value_active_vs_zero', np.nan),
                        'p_value_sham_vs_zero': coeff_data.get('p_value_sham_vs_zero', np.nan)
                    }
                    baseline_corrected_results.append(result)
    
    df_standard = pd.DataFrame(standard_results)
    df_baseline_corrected = pd.DataFrame(baseline_corrected_results)
    
    print(f"\n=== COMPARISON OF ANALYSIS METHODS ===")
    
    # Overall statistics
    print(f"\nStandard Analysis (Active vs Sham at each period):")
    print(f"  Total tests: {len(df_standard)}")
    standard_sig = len(df_standard[df_standard['p_value'] < 0.05])
    standard_sig_fdr = len(df_standard[df_standard['p_value_fdr'] < 0.05])
    print(f"  Significant (p < 0.05): {standard_sig} ({standard_sig/len(df_standard)*100:.1f}%)")
    print(f"  Significant (FDR corrected): {standard_sig_fdr}")
    
    print(f"\nBaseline-Corrected Analysis (Change scores: Active vs Sham):")
    print(f"  Total tests: {len(df_baseline_corrected)}")
    bc_sig = len(df_baseline_corrected[df_baseline_corrected['p_value'] < 0.05])
    bc_sig_fdr = len(df_baseline_corrected[df_baseline_corrected['p_value_fdr'] < 0.05])
    print(f"  Significant (p < 0.05): {bc_sig} ({bc_sig/len(df_baseline_corrected)*100:.1f}%)")
    print(f"  Significant (FDR corrected): {bc_sig_fdr}")
    
    # Period/comparison-specific breakdown
    print(f"\n=== BREAKDOWN BY PERIOD/COMPARISON ===")
    
    print(f"\nStandard Analysis by Period:")
    for period in ['baseline', 'stimulation', 'recovery']:
        period_data = df_standard[df_standard['comparison'] == period]
        sig_count = len(period_data[period_data['p_value'] < 0.05])
        print(f"  {period}: {sig_count}/{len(period_data)} significant ({sig_count/len(period_data)*100:.1f}%)")
    
    print(f"\nBaseline-Corrected Analysis by Comparison:")
    for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
        comp_data = df_baseline_corrected[df_baseline_corrected['comparison'] == comparison]
        sig_count = len(comp_data[comp_data['p_value'] < 0.05])
        print(f"  {comparison}: {sig_count}/{len(comp_data)} significant ({sig_count/len(comp_data)*100:.1f}%)")
    
    # Show most significant results from each method
    print(f"\n=== TOP SIGNIFICANT RESULTS ===")
    
    print(f"\nTop 10 Standard Analysis Results:")
    top_standard = df_standard.nsmallest(10, 'p_value')
    for i, (_, row) in enumerate(top_standard.iterrows(), 1):
        print(f"{i:2d}. ROI {row['roi_index']:3d} ({row['comparison']}-{row['coefficient']}): "
              f"p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
    
    print(f"\nTop 10 Baseline-Corrected Results:")
    top_bc = df_baseline_corrected.nsmallest(10, 'p_value')
    for i, (_, row) in enumerate(top_bc.iterrows(), 1):
        print(f"{i:2d}. ROI {row['roi_index']:3d} ({row['comparison']}-{row['coefficient']}): "
              f"p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
        print(f"     Active change: {row['active_change_mean']:.4f}, "
              f"Sham change: {row['sham_change_mean']:.4f}")
    
    # Compare overlapping significant results
    print(f"\n=== COMPARISON OF METHODS ===")
    
    # Map baseline-corrected comparisons to standard periods
    comparison_mapping = {
        'stimulation_minus_baseline': 'stimulation',
        'recovery_minus_baseline': 'recovery'
    }
    
    # Find overlapping significant results
    standard_sig_set = set()
    for _, row in df_standard[df_standard['p_value'] < 0.05].iterrows():
        standard_sig_set.add((row['roi_index'], row['comparison'], row['coefficient']))
    
    bc_sig_set = set()
    overlapping_sig = set()
    for _, row in df_baseline_corrected[df_baseline_corrected['p_value'] < 0.05].iterrows():
        bc_key = (row['roi_index'], row['comparison'], row['coefficient'])
        bc_sig_set.add(bc_key)
        
        # Check if corresponding standard result is also significant
        standard_period = comparison_mapping[row['comparison']]
        standard_key = (row['roi_index'], standard_period, row['coefficient'])
        if standard_key in standard_sig_set:
            overlapping_sig.add((row['roi_index'], row['comparison'], row['coefficient']))
    
    print(f"Significant results overlap:")
    print(f"  Standard only: {len(standard_sig_set - {(roi, comparison_mapping.get(comp, comp), coeff) for roi, comp, coeff in bc_sig_set})}")
    print(f"  Baseline-corrected only: {len(bc_sig_set - {(roi, comp.replace('_minus_baseline', ''), coeff) for roi, comp, coeff in standard_sig_set if comp.endswith('_minus_baseline')})}")
    print(f"  Both methods: {len(overlapping_sig)}")
    
    # Analysis of change directions
    print(f"\n=== ANALYSIS OF CHANGE DIRECTIONS ===")
    
    # Look at whether changes are in expected directions
    for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
        comp_data = df_baseline_corrected[df_baseline_corrected['comparison'] == comparison]
        
        # Count significant changes vs zero for each condition
        active_increases = len(comp_data[(comp_data['p_value_active_vs_zero'] < 0.05) & 
                                        (comp_data['active_change_mean'] > 0)])
        active_decreases = len(comp_data[(comp_data['p_value_active_vs_zero'] < 0.05) & 
                                        (comp_data['active_change_mean'] < 0)])
        
        sham_increases = len(comp_data[(comp_data['p_value_sham_vs_zero'] < 0.05) & 
                                      (comp_data['sham_change_mean'] > 0)])
        sham_decreases = len(comp_data[(comp_data['p_value_sham_vs_zero'] < 0.05) & 
                                      (comp_data['sham_change_mean'] < 0)])
        
        print(f"\n{comparison}:")
        print(f"  Active condition changes vs zero (p<0.05): {active_increases} increases, {active_decreases} decreases")
        print(f"  Sham condition changes vs zero (p<0.05): {sham_increases} increases, {sham_decreases} decreases")
    
    # Save detailed comparison results
    output_path = "/Users/jacekdmochowski/PROJECTS/fus_bold/code/analysis_method_comparison.csv"
    
    # Combine results with method identifier
    df_combined = pd.concat([df_standard, df_baseline_corrected], ignore_index=True)
    df_combined.to_csv(output_path, index=False)
    print(f"\nDetailed comparison results saved to: analysis_method_comparison.csv")
    
    return df_standard, df_baseline_corrected

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python compare_analysis_methods.py <baseline_corrected_json_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    df_standard, df_baseline_corrected = compare_analysis_methods(json_file)

if __name__ == "__main__":
    main()