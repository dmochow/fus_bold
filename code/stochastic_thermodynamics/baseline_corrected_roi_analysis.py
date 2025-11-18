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

def calculate_baseline_corrected_changes(df_roi):
    """Calculate baseline-corrected changes for a single ROI."""
    
    results = {}
    
    # Get baseline, stimulation, and recovery data for each condition
    conditions = ['active', 'sham']
    periods = ['baseline', 'stimulation', 'recovery']
    coefficients = ['drift_0', 'drift_1', 'diffusion']
    
    # Organize data by condition and period
    condition_data = {}
    for condition in conditions:
        condition_data[condition] = {}
        for period in periods:
            period_data = df_roi[(df_roi['condition'] == condition) & (df_roi['period'] == period)]
            condition_data[condition][period] = period_data
    
    # Calculate baseline-corrected changes
    change_comparisons = [
        ('stimulation_minus_baseline', 'stimulation', 'baseline'),
        ('recovery_minus_baseline', 'recovery', 'baseline')
    ]
    
    for comparison_name, target_period, reference_period in change_comparisons:
        comparison_results = {
            'comparison_type': comparison_name,
            'coefficients': {}
        }
        
        # Calculate changes for each coefficient
        for coeff in coefficients:
            try:
                # Calculate changes for each condition
                active_changes = []
                sham_changes = []
                
                # For active condition
                active_baseline = condition_data['active'][reference_period]
                active_target = condition_data['active'][target_period]
                
                if len(active_baseline) > 0 and len(active_target) > 0:
                    # Match subjects between periods
                    common_subjects = set(active_baseline['subject'].values) & set(active_target['subject'].values)
                    
                    for subject in common_subjects:
                        baseline_val = active_baseline[active_baseline['subject'] == subject][coeff].iloc[0]
                        target_val = active_target[active_target['subject'] == subject][coeff].iloc[0]
                        active_changes.append(target_val - baseline_val)
                
                # For sham condition
                sham_baseline = condition_data['sham'][reference_period]
                sham_target = condition_data['sham'][target_period]
                
                if len(sham_baseline) > 0 and len(sham_target) > 0:
                    # Match subjects between periods
                    common_subjects = set(sham_baseline['subject'].values) & set(sham_target['subject'].values)
                    
                    for subject in common_subjects:
                        baseline_val = sham_baseline[sham_baseline['subject'] == subject][coeff].iloc[0]
                        target_val = sham_target[sham_target['subject'] == subject][coeff].iloc[0]
                        sham_changes.append(target_val - baseline_val)
                
                # Perform statistical test on changes
                if len(active_changes) > 1 and len(sham_changes) > 1:
                    active_changes = np.array(active_changes)
                    sham_changes = np.array(sham_changes)
                    
                    # T-test comparing active vs sham changes
                    t_stat, p_val = stats.ttest_ind(active_changes, sham_changes)
                    
                    # Effect size
                    pooled_std = np.sqrt(((len(active_changes) - 1) * np.var(active_changes, ddof=1) + 
                                         (len(sham_changes) - 1) * np.var(sham_changes, ddof=1)) / 
                                        (len(active_changes) + len(sham_changes) - 2))
                    
                    cohens_d = (np.mean(active_changes) - np.mean(sham_changes)) / pooled_std if pooled_std > 0 else 0
                    
                    # One-sample t-tests (testing if changes are different from zero)
                    t_active_vs_zero, p_active_vs_zero = stats.ttest_1samp(active_changes, 0)
                    t_sham_vs_zero, p_sham_vs_zero = stats.ttest_1samp(sham_changes, 0)
                    
                    comparison_results['coefficients'][coeff] = {
                        'n_active_subjects': len(active_changes),
                        'n_sham_subjects': len(sham_changes),
                        'active_change_mean': float(np.mean(active_changes)),
                        'active_change_std': float(np.std(active_changes, ddof=1)),
                        'active_change_sem': float(np.std(active_changes, ddof=1) / np.sqrt(len(active_changes))),
                        'sham_change_mean': float(np.mean(sham_changes)),
                        'sham_change_std': float(np.std(sham_changes, ddof=1)),
                        'sham_change_sem': float(np.std(sham_changes, ddof=1) / np.sqrt(len(sham_changes))),
                        'change_difference': float(np.mean(active_changes) - np.mean(sham_changes)),
                        
                        # Active vs Sham comparison
                        't_statistic_active_vs_sham': float(t_stat),
                        'p_value_active_vs_sham': float(p_val),
                        'cohens_d_active_vs_sham': float(cohens_d),
                        
                        # One-sample tests (changes vs zero)
                        't_statistic_active_vs_zero': float(t_active_vs_zero),
                        'p_value_active_vs_zero': float(p_active_vs_zero),
                        't_statistic_sham_vs_zero': float(t_sham_vs_zero),
                        'p_value_sham_vs_zero': float(p_sham_vs_zero),
                        
                        'degrees_of_freedom': int(len(active_changes) + len(sham_changes) - 2)
                    }
                    
                else:
                    comparison_results['coefficients'][coeff] = {'error': 'insufficient_subjects_for_changes'}
                    
            except Exception as e:
                comparison_results['coefficients'][coeff] = {'error': str(e)}
        
        results[comparison_name] = comparison_results
    
    return results

def process_roi_baseline_corrected(df_roi):
    """Process both standard and baseline-corrected statistics for a single ROI."""
    
    # First get standard statistics (original approach)
    standard_results = {}
    for period in ['baseline', 'stimulation', 'recovery']:
        period_data = df_roi[df_roi['period'] == period]
        active_data = period_data[period_data['condition'] == 'active']
        sham_data = period_data[period_data['condition'] == 'sham']
        
        period_results = {
            'n_active': len(active_data),
            'n_sham': len(sham_data),
            'coefficients': {}
        }
        
        if len(active_data) > 1 and len(sham_data) > 1:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                try:
                    active_vals = active_data[coeff].values
                    sham_vals = sham_data[coeff].values
                    
                    t_stat, p_val = stats.ttest_ind(active_vals, sham_vals)
                    
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
        
        standard_results[period] = period_results
    
    # Get baseline-corrected statistics
    baseline_corrected_results = calculate_baseline_corrected_changes(df_roi)
    
    return {
        'standard_comparisons': standard_results,
        'baseline_corrected_comparisons': baseline_corrected_results
    }

def baseline_corrected_roi_analysis(csv_file, output_file, max_rois=None):
    """Perform both standard and baseline-corrected ROI analysis."""
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Data shape: {df.shape}")
    
    # Load labels
    labels = load_difumo_labels()
    
    # Get unique ROIs
    unique_rois = sorted(df['roi'].unique())
    if max_rois:
        unique_rois = unique_rois[:max_rois]
    
    print(f"Processing {len(unique_rois)} ROIs with baseline correction...")
    
    # Process each ROI
    roi_results = {}
    
    # For multiple comparison correction, collect p-values separately for each analysis type
    standard_p_values = {coeff: {period: [] for period in ['baseline', 'stimulation', 'recovery']} 
                        for coeff in ['drift_0', 'drift_1', 'diffusion']}
    standard_roi_mappings = {coeff: {period: [] for period in ['baseline', 'stimulation', 'recovery']} 
                            for coeff in ['drift_0', 'drift_1', 'diffusion']}
    
    baseline_corrected_p_values = {coeff: {comparison: [] for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']} 
                                  for coeff in ['drift_0', 'drift_1', 'diffusion']}
    baseline_corrected_roi_mappings = {coeff: {comparison: [] for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']} 
                                      for coeff in ['drift_0', 'drift_1', 'diffusion']}
    
    for roi_idx in tqdm(unique_rois, desc="Processing ROIs"):
        roi_data = df[df['roi'] == roi_idx]
        
        # Get label
        roi_label = labels[roi_idx] if roi_idx < len(labels) else f"ROI_{roi_idx}"
        
        # Process statistics
        roi_stats = process_roi_baseline_corrected(roi_data)
        
        roi_results[f"roi_{roi_idx:04d}"] = {
            'roi_index': int(roi_idx),
            'roi_label': roi_label,
            'analysis_results': roi_stats
        }
        
        # Collect p-values for multiple comparison correction
        
        # Standard comparisons
        for period in ['baseline', 'stimulation', 'recovery']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_stats['standard_comparisons'][period]['coefficients'][coeff]
                
                if 'p_value' in coeff_data and not np.isnan(coeff_data['p_value']):
                    standard_p_values[coeff][period].append(coeff_data['p_value'])
                    standard_roi_mappings[coeff][period].append(f"roi_{roi_idx:04d}")
        
        # Baseline-corrected comparisons
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_stats['baseline_corrected_comparisons'][comparison]['coefficients'][coeff]
                
                if 'p_value_active_vs_sham' in coeff_data and not np.isnan(coeff_data['p_value_active_vs_sham']):
                    baseline_corrected_p_values[coeff][comparison].append(coeff_data['p_value_active_vs_sham'])
                    baseline_corrected_roi_mappings[coeff][comparison].append(f"roi_{roi_idx:04d}")
    
    # Apply multiple comparison corrections
    print("Applying multiple comparison corrections...")
    
    # Standard comparisons corrections
    for coeff in ['drift_0', 'drift_1', 'diffusion']:
        for period in ['baseline', 'stimulation', 'recovery']:
            p_vals = standard_p_values[coeff][period]
            roi_keys = standard_roi_mappings[coeff][period]
            
            if len(p_vals) > 0:
                # Bonferroni and FDR corrections
                p_bonf = [min(p * len(p_vals), 1.0) for p in p_vals]
                
                sorted_indices = np.argsort(p_vals)
                p_fdr = [0] * len(p_vals)
                for i, idx in enumerate(sorted_indices):
                    p_fdr[idx] = min(p_vals[idx] * len(p_vals) / (i + 1), 1.0)
                
                # Add to results
                for i, roi_key in enumerate(roi_keys):
                    coeff_results = roi_results[roi_key]['analysis_results']['standard_comparisons'][period]['coefficients'][coeff]
                    coeff_results['p_value_bonferroni'] = float(p_bonf[i])
                    coeff_results['p_value_fdr'] = float(p_fdr[i])
    
    # Baseline-corrected comparisons corrections
    for coeff in ['drift_0', 'drift_1', 'diffusion']:
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            p_vals = baseline_corrected_p_values[coeff][comparison]
            roi_keys = baseline_corrected_roi_mappings[coeff][comparison]
            
            if len(p_vals) > 0:
                # Bonferroni and FDR corrections
                p_bonf = [min(p * len(p_vals), 1.0) for p in p_vals]
                
                sorted_indices = np.argsort(p_vals)
                p_fdr = [0] * len(p_vals)
                for i, idx in enumerate(sorted_indices):
                    p_fdr[idx] = min(p_vals[idx] * len(p_vals) / (i + 1), 1.0)
                
                # Add to results
                for i, roi_key in enumerate(roi_keys):
                    coeff_results = roi_results[roi_key]['analysis_results']['baseline_corrected_comparisons'][comparison]['coefficients'][coeff]
                    coeff_results['p_value_active_vs_sham_bonferroni'] = float(p_bonf[i])
                    coeff_results['p_value_active_vs_sham_fdr'] = float(p_fdr[i])
    
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
            'analysis_types': ['standard_comparisons', 'baseline_corrected_comparisons'],
            'baseline_corrected_comparisons': ['stimulation_minus_baseline', 'recovery_minus_baseline'],
            'statistical_tests': {
                'standard': 'independent_t_test (active vs sham at each period)',
                'baseline_corrected': 'independent_t_test (active_change vs sham_change) and one_sample_t_test (change vs zero)'
            },
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
    standard_significant = 0
    baseline_corrected_significant = 0
    
    for roi_key, roi_data in roi_results.items():
        # Standard comparisons
        for period in ['baseline', 'stimulation', 'recovery']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results']['standard_comparisons'][period]['coefficients'][coeff]
                if coeff_data.get('p_value', 1.0) < 0.05:
                    standard_significant += 1
        
        # Baseline-corrected comparisons
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results']['baseline_corrected_comparisons'][comparison]['coefficients'][coeff]
                if coeff_data.get('p_value_active_vs_sham', 1.0) < 0.05:
                    baseline_corrected_significant += 1
    
    standard_total = len(unique_rois) * 3 * 3  # ROIs * periods * coefficients
    baseline_corrected_total = len(unique_rois) * 2 * 3  # ROIs * comparisons * coefficients
    
    print(f"\nQuick Summary:")
    print(f"  Standard comparisons (active vs sham at each period):")
    print(f"    Total tests: {standard_total}")
    print(f"    Significant (uncorrected, p<0.05): {standard_significant}")
    print(f"  Baseline-corrected comparisons (change scores):")
    print(f"    Total tests: {baseline_corrected_total}")
    print(f"    Significant (uncorrected, p<0.05): {baseline_corrected_significant}")
    
    return output_data

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Baseline-corrected ROI-specific Fokker-Planck analysis')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with results')
    parser.add_argument('--output', type=str, default='baseline_corrected_roi_results.json',
                       help='Output JSON filename')
    parser.add_argument('--max-rois', type=int, default=None,
                       help='Maximum number of ROIs to process (for testing)')
    
    args = parser.parse_args()
    
    # Run analysis
    output_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/{args.output}"
    results = baseline_corrected_roi_analysis(args.input, output_path, args.max_rois)

if __name__ == "__main__":
    main()