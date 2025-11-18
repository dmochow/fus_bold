import json
import pandas as pd
import numpy as np

def summarize_roi_findings(json_file, output_csv=None):
    """Create a comprehensive summary of ROI findings."""
    
    print(f"Loading results from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    analysis_info = data['analysis_info']
    roi_results = data['roi_results']
    
    print(f"Analysis info:")
    print(f"  Subjects: {analysis_info['n_subjects']}")
    print(f"  ROIs processed: {analysis_info['n_rois_processed']}")
    
    # Create comprehensive results table
    results_list = []
    
    for roi_key, roi_data in roi_results.items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for period in ['baseline', 'stimulation', 'recovery']:
            period_data = roi_data['periods'][period]
            
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = period_data['coefficients'][coeff]
                
                if 'p_value' in coeff_data:
                    result = {
                        'roi_index': roi_idx,
                        'roi_label': roi_label,
                        'period': period,
                        'coefficient': coeff,
                        'active_mean': coeff_data.get('active_mean', np.nan),
                        'active_std': coeff_data.get('active_std', np.nan),
                        'sham_mean': coeff_data.get('sham_mean', np.nan),
                        'sham_std': coeff_data.get('sham_std', np.nan),
                        'mean_difference': coeff_data.get('mean_difference', np.nan),
                        't_statistic': coeff_data.get('t_statistic', np.nan),
                        'p_value': coeff_data.get('p_value', np.nan),
                        'p_value_fdr': coeff_data.get('p_value_fdr', np.nan),
                        'p_value_bonferroni': coeff_data.get('p_value_bonferroni', np.nan),
                        'cohens_d': coeff_data.get('cohens_d', np.nan)
                    }
                    results_list.append(result)
    
    df_results = pd.DataFrame(results_list)
    
    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total ROI-period-coefficient combinations: {len(df_results)}")
    
    # Significant results by correction method
    sig_uncorrected = len(df_results[df_results['p_value'] < 0.05])
    sig_fdr = len(df_results[df_results['p_value_fdr'] < 0.05])
    sig_bonf = len(df_results[df_results['p_value_bonferroni'] < 0.05])
    
    print(f"Significant results (p < 0.05):")
    print(f"  Uncorrected: {sig_uncorrected} ({sig_uncorrected/len(df_results)*100:.1f}%)")
    print(f"  FDR corrected: {sig_fdr} ({sig_fdr/len(df_results)*100:.1f}%)")
    print(f"  Bonferroni corrected: {sig_bonf} ({sig_bonf/len(df_results)*100:.1f}%)")
    
    # Effect sizes
    print(f"\nEffect sizes (Cohen's d):")
    print(f"  Mean: {df_results['cohens_d'].mean():.3f}")
    print(f"  Std: {df_results['cohens_d'].std():.3f}")
    print(f"  Large effects (|d| > 0.8): {len(df_results[abs(df_results['cohens_d']) > 0.8])}")
    print(f"  Medium effects (0.5 < |d| ≤ 0.8): {len(df_results[(abs(df_results['cohens_d']) > 0.5) & (abs(df_results['cohens_d']) <= 0.8)])}")
    
    # Period-specific analysis
    print(f"\n=== PERIOD-SPECIFIC EFFECTS ===")
    for period in ['baseline', 'stimulation', 'recovery']:
        period_data = df_results[df_results['period'] == period]
        sig_count = len(period_data[period_data['p_value'] < 0.05])
        print(f"{period.capitalize()}: {sig_count} significant results")
        
        if sig_count > 0:
            # Top effects
            top_effects = period_data.nsmallest(3, 'p_value')
            for _, row in top_effects.iterrows():
                print(f"  ROI {row['roi_index']:3d} ({row['coefficient']}): "
                      f"p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
    
    # Coefficient-specific analysis
    print(f"\n=== COEFFICIENT-SPECIFIC EFFECTS ===")
    for coeff in ['drift_0', 'drift_1', 'diffusion']:
        coeff_data = df_results[df_results['coefficient'] == coeff]
        sig_count = len(coeff_data[coeff_data['p_value'] < 0.05])
        print(f"{coeff}: {sig_count} significant results")
        
        if sig_count > 0:
            print(f"  Mean effect size: {coeff_data[coeff_data['p_value'] < 0.05]['cohens_d'].mean():.3f}")
    
    # ROI-specific hotspots
    print(f"\n=== ROI HOTSPOTS (Most significant ROIs) ===")
    roi_sig_counts = df_results[df_results['p_value'] < 0.05].groupby('roi_index').size().sort_values(ascending=False)
    
    if len(roi_sig_counts) > 0:
        print("ROIs with most significant effects:")
        for roi_idx, count in roi_sig_counts.head(10).items():
            roi_label = df_results[df_results['roi_index'] == roi_idx]['roi_label'].iloc[0]
            print(f"  ROI {roi_idx:3d}: {count} significant effects - {roi_label[:50]}...")
    
    # Network analysis (if network info available)
    print(f"\n=== BRAIN NETWORK ANALYSIS ===")
    network_keywords = ['Default', 'Salience', 'Control', 'DorsAttn', 'SomMot', 'Vis', 'Limbic']
    
    for keyword in network_keywords:
        network_rois = df_results[df_results['roi_label'].str.contains(keyword, case=False, na=False)]
        if len(network_rois) > 0:
            sig_count = len(network_rois[network_rois['p_value'] < 0.05])
            print(f"{keyword} network: {sig_count}/{len(network_rois)} significant ({sig_count/len(network_rois)*100:.1f}%)")
    
    # Target region analysis (subgenual ACC)
    print(f"\n=== TARGET REGION ANALYSIS ===")
    target_keywords = ['cingulate', 'anterior cingulate', 'subgenual', 'ACC']
    
    for keyword in target_keywords:
        target_rois = df_results[df_results['roi_label'].str.contains(keyword, case=False, na=False)]
        if len(target_rois) > 0:
            sig_count = len(target_rois[target_rois['p_value'] < 0.05])
            print(f"ROIs containing '{keyword}': {len(target_rois)} ROIs, {sig_count} significant")
            
            if sig_count > 0:
                sig_target = target_rois[target_rois['p_value'] < 0.05].nsmallest(3, 'p_value')
                for _, row in sig_target.iterrows():
                    print(f"  ROI {row['roi_index']:3d} ({row['period']}-{row['coefficient']}): "
                          f"p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
    
    # Save detailed results
    if output_csv:
        csv_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/{output_csv}"
        df_results.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")
    
    # Create summary of most significant results
    most_significant = df_results.nsmallest(20, 'p_value')[
        ['roi_index', 'roi_label', 'period', 'coefficient', 'mean_difference', 
         'p_value', 'p_value_fdr', 'cohens_d']
    ]
    
    summary_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/top_significant_rois.csv"
    most_significant.to_csv(summary_path, index=False)
    print(f"Top 20 significant results saved to: top_significant_rois.csv")
    
    return df_results

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python summarize_roi_findings.py <json_results_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_csv = json_file.replace('.json', '_detailed_results.csv')
    
    df_results = summarize_roi_findings(json_file, output_csv)

if __name__ == "__main__":
    main()