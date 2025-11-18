import numpy as np
import pandas as pd
import json
from scipy import stats
from nilearn import datasets
import argparse
import warnings

def load_difumo_labels():
    """Load DiFuMo atlas labels."""
    print("Loading DiFuMo atlas labels...")
    difumo = datasets.fetch_atlas_difumo(dimension=1024)
    labels = difumo.labels
    print(f"Loaded {len(labels)} ROI labels")
    return labels

def roi_specific_statistical_analysis(df, labels):
    """Perform statistical analysis separately for each ROI."""
    
    print("Performing ROI-specific statistical analysis...")
    
    # Get unique ROIs in the data
    rois_in_data = sorted(df['roi'].unique())
    print(f"ROIs in data: {len(rois_in_data)} (expected: {len(labels)})")
    
    results = {}
    
    # Process each ROI
    for roi_idx in rois_in_data:
        roi_data = df[df['roi'] == roi_idx]
        
        # Get ROI label and convert to string
        if roi_idx < len(labels):
            roi_label = str(labels[roi_idx])
        else:
            roi_label = f"ROI_{roi_idx}"
        
        print(f"Processing ROI {roi_idx}: {roi_label}")
        
        roi_results = {
            'roi_index': int(roi_idx),
            'roi_label': roi_label,
            'periods': {}
        }
        
        # Analyze each period
        for period in ['baseline', 'stimulation', 'recovery']:
            period_data = roi_data[roi_data['period'] == period]
            
            active_data = period_data[period_data['condition'] == 'active']
            sham_data = period_data[period_data['condition'] == 'sham']
            
            period_results = {
                'n_active_subjects': len(active_data),
                'n_sham_subjects': len(sham_data),
                'coefficients': {}
            }
            
            if len(active_data) > 0 and len(sham_data) > 0:
                # Test each coefficient
                for coeff in ['drift_0', 'drift_1', 'diffusion']:
                    active_vals = active_data[coeff].values
                    sham_vals = sham_data[coeff].values
                    
                    # Basic statistics
                    active_mean = np.mean(active_vals)
                    active_std = np.std(active_vals, ddof=1)
                    sham_mean = np.mean(sham_vals)
                    sham_std = np.std(sham_vals, ddof=1)
                    
                    # Independent t-test
                    try:
                        t_stat, p_val = stats.ttest_ind(active_vals, sham_vals)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(active_vals) - 1) * np.var(active_vals, ddof=1) + 
                                             (len(sham_vals) - 1) * np.var(sham_vals, ddof=1)) / 
                                            (len(active_vals) + len(sham_vals) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (active_mean - sham_mean) / pooled_std
                        else:
                            cohens_d = np.nan
                        
                        # Confidence interval for difference in means
                        se_diff = pooled_std * np.sqrt(1/len(active_vals) + 1/len(sham_vals))
                        df_t = len(active_vals) + len(sham_vals) - 2
                        t_critical = stats.t.ppf(0.975, df_t)  # 95% CI
                        mean_diff = active_mean - sham_mean
                        ci_lower = mean_diff - t_critical * se_diff
                        ci_upper = mean_diff + t_critical * se_diff
                        
                    except Exception as e:
                        print(f"    Warning: Statistical test failed for {coeff}: {e}")
                        t_stat = np.nan
                        p_val = np.nan
                        cohens_d = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan
                        mean_diff = active_mean - sham_mean
                    
                    # Store results
                    period_results['coefficients'][coeff] = {
                        'active_mean': float(active_mean),
                        'active_std': float(active_std),
                        'active_sem': float(active_std / np.sqrt(len(active_vals))),
                        'sham_mean': float(sham_mean),
                        'sham_std': float(sham_std),
                        'sham_sem': float(sham_std / np.sqrt(len(sham_vals))),
                        'mean_difference': float(mean_diff),
                        'ci_95_lower': float(ci_lower),
                        'ci_95_upper': float(ci_upper),
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'cohens_d': float(cohens_d),
                        'degrees_of_freedom': int(len(active_vals) + len(sham_vals) - 2)
                    }
            
            else:
                print(f"    Warning: Insufficient data for {period} period")
                period_results['coefficients'] = {
                    coeff: {'error': 'insufficient_data'} 
                    for coeff in ['drift_0', 'drift_1', 'diffusion']
                }
            
            roi_results['periods'][period] = period_results
        
        results[f"roi_{roi_idx:04d}"] = roi_results
    
    return results

def add_multiple_comparison_corrections(results):
    """Add multiple comparison corrections to the results."""
    
    print("Adding multiple comparison corrections...")
    
    # Collect all p-values for each coefficient and period
    all_p_values = {}
    roi_period_coeff_mapping = {}
    
    for coeff in ['drift_0', 'drift_1', 'diffusion']:
        for period in ['baseline', 'stimulation', 'recovery']:
            key = f"{coeff}_{period}"
            all_p_values[key] = []
            roi_period_coeff_mapping[key] = []
            
            for roi_key, roi_data in results.items():
                period_data = roi_data['periods'].get(period, {})
                coeff_data = period_data.get('coefficients', {}).get(coeff, {})
                
                if 'p_value' in coeff_data and not np.isnan(coeff_data['p_value']):
                    all_p_values[key].append(coeff_data['p_value'])
                    roi_period_coeff_mapping[key].append(roi_key)
    
    # Apply corrections
    for key, p_vals in all_p_values.items():
        if len(p_vals) > 0:
            # Bonferroni correction
            p_bonf = [min(p * len(p_vals), 1.0) for p in p_vals]
            
            # FDR correction (Benjamini-Hochberg)
            try:
                from statsmodels.stats.multitest import multipletests
                reject_fdr, p_fdr, _, _ = multipletests(p_vals, method='fdr_bh')
            except ImportError:
                # Simple FDR approximation if statsmodels not available
                sorted_indices = np.argsort(p_vals)
                p_fdr = [np.nan] * len(p_vals)
                for i, idx in enumerate(sorted_indices):
                    p_fdr[idx] = min(p_vals[idx] * len(p_vals) / (i + 1), 1.0)
            
            # Add corrections to results
            parts = key.split('_')
            coeff = parts[0] + '_' + parts[1] if parts[0] in ['drift'] else parts[0]
            period = '_'.join(parts[2:]) if parts[0] in ['drift'] else '_'.join(parts[1:])
            
            for i, roi_key in enumerate(roi_period_coeff_mapping[key]):
                coeff_results = results[roi_key]['periods'][period]['coefficients'][coeff]
                coeff_results['p_value_bonferroni'] = float(p_bonf[i])
                coeff_results['p_value_fdr'] = float(p_fdr[i])
    
    return results

def summarize_significant_results(results, alpha=0.05, correction='fdr'):
    """Summarize significant results across all ROIs."""
    
    print(f"\nSummarizing significant results (α = {alpha}, correction = {correction})...")
    
    p_key = f'p_value_{correction}' if correction != 'uncorrected' else 'p_value'
    
    significant_results = []
    
    for roi_key, roi_data in results.items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for period in ['baseline', 'stimulation', 'recovery']:
            period_data = roi_data['periods'].get(period, {})
            
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = period_data.get('coefficients', {}).get(coeff, {})
                
                if p_key in coeff_data:
                    p_val = coeff_data[p_key]
                    if p_val < alpha:
                        significant_results.append({
                            'roi_index': roi_idx,
                            'roi_label': roi_label,
                            'period': period,
                            'coefficient': coeff,
                            'p_value': coeff_data['p_value'],
                            f'p_value_{correction}': p_val,
                            'cohens_d': coeff_data.get('cohens_d', np.nan),
                            'mean_difference': coeff_data.get('mean_difference', np.nan)
                        })
    
    print(f"Found {len(significant_results)} significant results")
    
    if len(significant_results) > 0:
        # Sort by p-value
        significant_results.sort(key=lambda x: x[f'p_value_{correction}'])
        
        print(f"\nTop 10 most significant results:")
        for i, result in enumerate(significant_results[:10]):
            print(f"{i+1:2d}. ROI {result['roi_index']:3d} ({result['roi_label'][:40]}...)")
            print(f"     {result['period']} - {result['coefficient']}: "
                  f"p={result[f'p_value_{correction}']:.4f}, d={result['cohens_d']:.3f}")
    
    return significant_results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='ROI-specific Fokker-Planck statistical analysis')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with Fokker-Planck results')
    parser.add_argument('--output', type=str, default='roi_statistical_analysis.json',
                       help='Output JSON filename')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Data shape: {df.shape}")
    
    # Load atlas labels
    labels = load_difumo_labels()
    
    # Perform ROI-specific analysis
    results = roi_specific_statistical_analysis(df, labels)
    
    # Add multiple comparison corrections
    results = add_multiple_comparison_corrections(results)
    
    # Add metadata
    metadata = {
        'analysis_info': {
            'input_file': args.input,
            'n_subjects': int(df['subject'].nunique()),
            'n_rois': int(df['roi'].nunique()),
            'conditions': list(df['condition'].unique()),
            'periods': list(df['period'].unique()),
            'coefficients': ['drift_0', 'drift_1', 'diffusion'],
            'statistical_test': 'independent_t_test',
            'multiple_comparison_corrections': ['bonferroni', 'fdr_bh'],
            'alpha_level': args.alpha
        },
        'roi_results': results
    }
    
    # Save results
    output_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/{args.output}"
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Summarize significant results
    significant_uncorrected = summarize_significant_results(results, args.alpha, 'uncorrected')
    significant_fdr = summarize_significant_results(results, args.alpha, 'fdr')
    significant_bonf = summarize_significant_results(results, args.alpha, 'bonferroni')
    
    print(f"\nSummary:")
    print(f"  Significant results (uncorrected): {len(significant_uncorrected)}")
    print(f"  Significant results (FDR): {len(significant_fdr)}")
    print(f"  Significant results (Bonferroni): {len(significant_bonf)}")
    
    # Save summary of significant results
    summary_output = args.output.replace('.json', '_significant_summary.json')
    summary_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/{summary_output}"
    
    summary_data = {
        'uncorrected': significant_uncorrected,
        'fdr_corrected': significant_fdr,
        'bonferroni_corrected': significant_bonf
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary of significant results saved to: {summary_path}")

if __name__ == "__main__":
    main()