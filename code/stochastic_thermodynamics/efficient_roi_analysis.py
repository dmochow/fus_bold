import numpy as np
import pandas as pd
import json
from scipy import stats
from nilearn import datasets
import argparse
from tqdm import tqdm

def load_difumo_labels():
    """Load DiFuMo atlas labels efficiently."""
    print("Loading DiFuMo atlas labels...")
    difumo = datasets.fetch_atlas_difumo(dimension=1024)
    # Convert labels to simple strings to avoid JSON serialization issues
    labels = [str(label).split("'")[1] if "'" in str(label) else str(label) for label in difumo.labels]
    print(f"Loaded {len(labels)} ROI labels")
    return labels

def process_roi_statistics(df_roi):
    """Process statistics for a single ROI."""
    results = {}
    
    for period in ['baseline', 'stimulation', 'recovery']:
        period_data = df_roi[df_roi['period'] == period]
        active_data = period_data[period_data['condition'] == 'active']
        sham_data = period_data[period_data['condition'] == 'sham']
        
        period_results = {
            'n_active': len(active_data),
            'n_sham': len(sham_data),
            'coefficients': {}
        }
        
        if len(active_data) > 1 and len(sham_data) > 1:  # Need at least 2 subjects per condition
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                try:
                    active_vals = active_data[coeff].values
                    sham_vals = sham_data[coeff].values
                    
                    # T-test
                    t_stat, p_val = stats.ttest_ind(active_vals, sham_vals)
                    
                    # Effect size
                    pooled_std = np.sqrt(((len(active_vals) - 1) * np.var(active_vals, ddof=1) + 
                                         (len(sham_vals) - 1) * np.var(sham_vals, ddof=1)) / 
                                        (len(active_vals) + len(sham_vals) - 2))
                    
                    cohens_d = (np.mean(active_vals) - np.mean(sham_vals)) / pooled_std if pooled_std > 0 else 0
                    
                    period_results['coefficients'][coeff] = {
                        'active_mean': float(np.mean(active_vals)),
                        'active_std': float(np.std(active_vals, ddof=1)),
                        'sham_mean': float(np.mean(sham_vals)),
                        'sham_std': float(np.std(sham_vals, ddof=1)),
                        'mean_difference': float(np.mean(active_vals) - np.mean(sham_vals)),
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'cohens_d': float(cohens_d)
                    }
                    
                except Exception as e:
                    period_results['coefficients'][coeff] = {'error': str(e)}
        else:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                period_results['coefficients'][coeff] = {'error': 'insufficient_subjects'}
        
        results[period] = period_results
    
    return results

def efficient_roi_analysis(csv_file, output_file, max_rois=None):
    """Efficient ROI-specific analysis that can handle large datasets."""
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Data shape: {df.shape}")
    
    # Load labels
    labels = load_difumo_labels()
    
    # Get unique ROIs
    unique_rois = sorted(df['roi'].unique())
    if max_rois:
        unique_rois = unique_rois[:max_rois]
    
    print(f"Processing {len(unique_rois)} ROIs...")
    
    # Process each ROI
    roi_results = {}
    all_p_values = {coeff: {period: [] for period in ['baseline', 'stimulation', 'recovery']} 
                    for coeff in ['drift_0', 'drift_1', 'diffusion']}
    roi_mappings = {coeff: {period: [] for period in ['baseline', 'stimulation', 'recovery']} 
                    for coeff in ['drift_0', 'drift_1', 'diffusion']}
    
    for roi_idx in tqdm(unique_rois, desc="Processing ROIs"):
        roi_data = df[df['roi'] == roi_idx]
        
        # Get label
        roi_label = labels[roi_idx] if roi_idx < len(labels) else f"ROI_{roi_idx}"
        
        # Process statistics
        roi_stats = process_roi_statistics(roi_data)
        
        roi_results[f"roi_{roi_idx:04d}"] = {
            'roi_index': int(roi_idx),
            'roi_label': roi_label,
            'periods': roi_stats
        }
        
        # Collect p-values for multiple comparison correction
        for period in ['baseline', 'stimulation', 'recovery']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                period_data = roi_stats.get(period, {})
                coeff_data = period_data.get('coefficients', {}).get(coeff, {})
                
                if 'p_value' in coeff_data and not np.isnan(coeff_data['p_value']):
                    all_p_values[coeff][period].append(coeff_data['p_value'])
                    roi_mappings[coeff][period].append(f"roi_{roi_idx:04d}")
    
    # Apply multiple comparison corrections
    print("Applying multiple comparison corrections...")
    
    for coeff in ['drift_0', 'drift_1', 'diffusion']:
        for period in ['baseline', 'stimulation', 'recovery']:
            p_vals = all_p_values[coeff][period]
            roi_keys = roi_mappings[coeff][period]
            
            if len(p_vals) > 0:
                # Bonferroni
                p_bonf = [min(p * len(p_vals), 1.0) for p in p_vals]
                
                # Simple FDR (Benjamini-Hochberg)
                sorted_indices = np.argsort(p_vals)
                p_fdr = [0] * len(p_vals)
                for i, idx in enumerate(sorted_indices):
                    p_fdr[idx] = min(p_vals[idx] * len(p_vals) / (i + 1), 1.0)
                
                # Add to results
                for i, roi_key in enumerate(roi_keys):
                    roi_results[roi_key]['periods'][period]['coefficients'][coeff]['p_value_bonferroni'] = float(p_bonf[i])
                    roi_results[roi_key]['periods'][period]['coefficients'][coeff]['p_value_fdr'] = float(p_fdr[i])
    
    # Create final output
    output_data = {
        'analysis_info': {
            'input_file': csv_file,
            'n_subjects': int(df['subject'].nunique()),
            'n_rois_processed': len(unique_rois),
            'n_rois_total': len(labels),
            'conditions': ['active', 'sham'],
            'periods': ['baseline', 'stimulation', 'recovery'],
            'coefficients': ['drift_0', 'drift_1', 'diffusion'],
            'statistical_test': 'independent_t_test',
            'multiple_corrections': ['bonferroni', 'fdr_bh']
        },
        'roi_results': roi_results
    }
    
    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Analysis complete!")
    
    # Quick summary
    significant_count = 0
    significant_fdr_count = 0
    
    for roi_key, roi_data in roi_results.items():
        for period in ['baseline', 'stimulation', 'recovery']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['periods'][period]['coefficients'][coeff]
                if 'p_value' in coeff_data:
                    if coeff_data['p_value'] < 0.05:
                        significant_count += 1
                    if coeff_data.get('p_value_fdr', 1.0) < 0.05:
                        significant_fdr_count += 1
    
    total_tests = len(unique_rois) * 3 * 3  # ROIs * periods * coefficients
    print(f"\nQuick Summary:")
    print(f"  Total statistical tests: {total_tests}")
    print(f"  Significant (uncorrected, p<0.05): {significant_count}")
    print(f"  Significant (FDR corrected, p<0.05): {significant_fdr_count}")
    
    return output_data

def find_significant_rois(json_file, alpha=0.05, correction='fdr'):
    """Find and summarize significant ROIs from the analysis."""
    
    print(f"Loading results from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    roi_results = data['roi_results']
    p_key = f'p_value_{correction}' if correction != 'uncorrected' else 'p_value'
    
    significant_results = []
    
    for roi_key, roi_data in roi_results.items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        for period in ['baseline', 'stimulation', 'recovery']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['periods'][period]['coefficients'][coeff]
                
                if p_key in coeff_data and coeff_data[p_key] < alpha:
                    significant_results.append({
                        'roi_index': roi_idx,
                        'roi_label': roi_label,
                        'period': period,
                        'coefficient': coeff,
                        'p_value': coeff_data.get('p_value', np.nan),
                        'p_value_corrected': coeff_data[p_key],
                        'cohens_d': coeff_data.get('cohens_d', np.nan),
                        'mean_difference': coeff_data.get('mean_difference', np.nan)
                    })
    
    # Sort by corrected p-value
    significant_results.sort(key=lambda x: x['p_value_corrected'])
    
    print(f"\nSignificant results (α={alpha}, {correction} correction): {len(significant_results)}")
    
    if len(significant_results) > 0:
        print("\nTop 10 most significant:")
        for i, result in enumerate(significant_results[:10]):
            print(f"{i+1:2d}. ROI {result['roi_index']:3d}: {result['roi_label'][:50]}...")
            print(f"    {result['period']} - {result['coefficient']}: "
                  f"p_corrected={result['p_value_corrected']:.4f}, d={result['cohens_d']:.3f}")
    
    return significant_results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Efficient ROI-specific Fokker-Planck analysis')
    parser.add_argument('--input', type=str, required=False,
                       help='Input CSV file with results')
    parser.add_argument('--output', type=str, default='roi_analysis_results.json',
                       help='Output JSON filename')
    parser.add_argument('--max-rois', type=int, default=None,
                       help='Maximum number of ROIs to process (for testing)')
    parser.add_argument('--find-significant', type=str, default=None,
                       help='Find significant results in existing JSON file')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level')
    parser.add_argument('--correction', type=str, default='fdr',
                       choices=['uncorrected', 'fdr', 'bonferroni'],
                       help='Multiple comparison correction method')
    
    args = parser.parse_args()
    
    if args.find_significant:
        # Just find significant results
        significant_results = find_significant_rois(args.find_significant, args.alpha, args.correction)
    else:
        # Run full analysis
        output_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/{args.output}"
        results = efficient_roi_analysis(args.input, output_path, args.max_rois)

if __name__ == "__main__":
    main()