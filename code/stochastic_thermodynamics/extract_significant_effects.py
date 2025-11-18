import json
import pandas as pd
import numpy as np
import argparse

def extract_significant_effects(json_file, alpha=0.05, correction='uncorrected', output_csv=None):
    """Extract all significant effects from baseline-corrected analysis."""
    
    print(f"Loading baseline-corrected analysis results from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    analysis_info = data['analysis_info']
    roi_results = data['roi_results']
    
    print(f"Analysis contains {analysis_info['n_rois_processed']} ROIs")
    print(f"Extracting effects with {correction} p-value < {alpha}")
    
    # Determine which p-value field to use
    if correction == 'uncorrected':
        p_value_field = 'p_value'
        standard_p_field = 'p_value'
        baseline_p_field = 'p_value_active_vs_sham'
    elif correction == 'fdr':
        p_value_field = 'p_value_fdr'
        standard_p_field = 'p_value_fdr'
        baseline_p_field = 'p_value_active_vs_sham_fdr'
    elif correction == 'bonferroni':
        p_value_field = 'p_value_bonferroni'
        standard_p_field = 'p_value_bonferroni'
        baseline_p_field = 'p_value_active_vs_sham_bonferroni'
    else:
        raise ValueError("correction must be 'uncorrected', 'fdr', or 'bonferroni'")
    
    significant_effects = []
    
    for roi_key, roi_data in roi_results.items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        # Extract standard comparison results (if significant)
        for period in ['baseline', 'stimulation', 'recovery']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results']['standard_comparisons'][period]['coefficients'][coeff]
                
                if standard_p_field in coeff_data:
                    p_val = coeff_data[standard_p_field]
                    if p_val < alpha:
                        effect = {
                            'analysis_type': 'standard',
                            'roi_index': roi_idx,
                            'roi_label': roi_label,
                            'comparison': period,
                            'coefficient': coeff,
                            'p_value': coeff_data.get('p_value', np.nan),
                            'p_value_fdr': coeff_data.get('p_value_fdr', np.nan),
                            'p_value_bonferroni': coeff_data.get('p_value_bonferroni', np.nan),
                            'p_value_used': p_val,
                            'cohens_d': coeff_data.get('cohens_d', np.nan),
                            'active_mean': coeff_data.get('active_mean', np.nan),
                            'active_std': coeff_data.get('active_std', np.nan),
                            'sham_mean': coeff_data.get('sham_mean', np.nan),
                            'sham_std': coeff_data.get('sham_std', np.nan),
                            'mean_difference': coeff_data.get('mean_difference', np.nan),
                            't_statistic': coeff_data.get('t_statistic', np.nan),
                            
                            # Baseline-corrected specific fields (empty for standard)
                            'active_change_mean': np.nan,
                            'active_change_std': np.nan,
                            'sham_change_mean': np.nan,
                            'sham_change_std': np.nan,
                            'change_difference': np.nan,
                            'p_value_active_vs_zero': np.nan,
                            'p_value_sham_vs_zero': np.nan
                        }
                        significant_effects.append(effect)
        
        # Extract baseline-corrected comparison results (if significant)
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results']['baseline_corrected_comparisons'][comparison]['coefficients'][coeff]
                
                if baseline_p_field in coeff_data:
                    p_val = coeff_data[baseline_p_field]
                    if p_val < alpha:
                        effect = {
                            'analysis_type': 'baseline_corrected',
                            'roi_index': roi_idx,
                            'roi_label': roi_label,
                            'comparison': comparison,
                            'coefficient': coeff,
                            'p_value': coeff_data.get('p_value_active_vs_sham', np.nan),
                            'p_value_fdr': coeff_data.get('p_value_active_vs_sham_fdr', np.nan),
                            'p_value_bonferroni': coeff_data.get('p_value_active_vs_sham_bonferroni', np.nan),
                            'p_value_used': p_val,
                            'cohens_d': coeff_data.get('cohens_d_active_vs_sham', np.nan),
                            't_statistic': coeff_data.get('t_statistic_active_vs_sham', np.nan),
                            
                            # Standard analysis fields (computed from changes)
                            'active_mean': coeff_data.get('active_change_mean', np.nan),
                            'active_std': coeff_data.get('active_change_std', np.nan),
                            'sham_mean': coeff_data.get('sham_change_mean', np.nan),
                            'sham_std': coeff_data.get('sham_change_std', np.nan),
                            'mean_difference': coeff_data.get('change_difference', np.nan),
                            
                            # Baseline-corrected specific fields
                            'active_change_mean': coeff_data.get('active_change_mean', np.nan),
                            'active_change_std': coeff_data.get('active_change_std', np.nan),
                            'sham_change_mean': coeff_data.get('sham_change_mean', np.nan),
                            'sham_change_std': coeff_data.get('sham_change_std', np.nan),
                            'change_difference': coeff_data.get('change_difference', np.nan),
                            'p_value_active_vs_zero': coeff_data.get('p_value_active_vs_zero', np.nan),
                            'p_value_sham_vs_zero': coeff_data.get('p_value_sham_vs_zero', np.nan)
                        }
                        significant_effects.append(effect)
    
    # Convert to DataFrame
    df_significant = pd.DataFrame(significant_effects)
    
    if len(df_significant) == 0:
        print(f"No significant effects found with {correction} correction at α = {alpha}")
        return df_significant
    
    # Sort by p-value
    df_significant = df_significant.sort_values('p_value_used').reset_index(drop=True)
    
    # Add effect size categories
    df_significant['effect_size_category'] = df_significant['cohens_d'].apply(
        lambda d: 'large' if abs(d) >= 0.8 else ('medium' if abs(d) >= 0.5 else ('small' if abs(d) >= 0.2 else 'negligible'))
    )
    
    # Add direction for baseline-corrected effects
    df_significant['active_change_direction'] = df_significant['active_change_mean'].apply(
        lambda x: 'increase' if x > 0 else ('decrease' if x < 0 else 'no_change') if not pd.isna(x) else 'N/A'
    )
    df_significant['sham_change_direction'] = df_significant['sham_change_mean'].apply(
        lambda x: 'increase' if x > 0 else ('decrease' if x < 0 else 'no_change') if not pd.isna(x) else 'N/A'
    )
    
    # Summary statistics
    print(f"\n=== SIGNIFICANT EFFECTS SUMMARY ===")
    print(f"Total significant effects: {len(df_significant)}")
    
    # By analysis type
    type_counts = df_significant['analysis_type'].value_counts()
    print(f"\nBy analysis type:")
    for analysis_type, count in type_counts.items():
        print(f"  {analysis_type}: {count}")
    
    # By comparison/period
    comparison_counts = df_significant['comparison'].value_counts()
    print(f"\nBy comparison/period:")
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
    
    # Top ROIs with most effects
    roi_counts = df_significant['roi_index'].value_counts().head(10)
    print(f"\nTop 10 ROIs with most significant effects:")
    for roi_idx, count in roi_counts.items():
        roi_label = df_significant[df_significant['roi_index'] == roi_idx]['roi_label'].iloc[0]
        print(f"  ROI {roi_idx:3d}: {count} effects - {roi_label[:50]}...")
    
    # For baseline-corrected effects, show change directions
    baseline_corrected = df_significant[df_significant['analysis_type'] == 'baseline_corrected']
    if len(baseline_corrected) > 0:
        print(f"\nBaseline-corrected effect directions:")
        print(f"  Active condition changes:")
        active_dir_counts = baseline_corrected['active_change_direction'].value_counts()
        for direction, count in active_dir_counts.items():
            print(f"    {direction}: {count}")
        
        print(f"  Sham condition changes:")
        sham_dir_counts = baseline_corrected['sham_change_direction'].value_counts()
        for direction, count in sham_dir_counts.items():
            print(f"    {direction}: {count}")
    
    # Save to CSV
    if output_csv is None:
        correction_suffix = f"_{correction}" if correction != 'uncorrected' else ""
        output_csv = f"significant_effects{correction_suffix}_alpha_{alpha:.3f}.csv"
    
    output_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/{output_csv}"
    df_significant.to_csv(output_path, index=False)
    print(f"\nSignificant effects saved to: {output_csv}")
    
    # Create separate files for each analysis type
    for analysis_type in df_significant['analysis_type'].unique():
        type_df = df_significant[df_significant['analysis_type'] == analysis_type]
        type_output = output_csv.replace('.csv', f'_{analysis_type}.csv')
        type_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/{type_output}"
        type_df.to_csv(type_path, index=False)
        print(f"{analysis_type} effects saved to: {type_output}")
    
    return df_significant

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Extract significant effects from baseline-corrected analysis')
    parser.add_argument('json_file', type=str,
                       help='Input JSON file with baseline-corrected analysis results')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    parser.add_argument('--correction', type=str, default='uncorrected',
                       choices=['uncorrected', 'fdr', 'bonferroni'],
                       help='Multiple comparison correction method (default: uncorrected)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    df_significant = extract_significant_effects(
        args.json_file, 
        alpha=args.alpha, 
        correction=args.correction, 
        output_csv=args.output
    )

if __name__ == "__main__":
    main()