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

def calculate_paired_baseline_corrected_changes(df_roi, test_type='paired'):
    """Calculate baseline-corrected changes for a single ROI with paired or unpaired tests."""
    
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
            'test_type': test_type,
            'coefficients': {}
        }
        
        # Calculate changes for each coefficient
        for coeff in coefficients:
            try:
                if test_type == 'paired':
                    # Paired analysis: match subjects between active and sham sessions
                    paired_changes = []
                    subject_info = []
                    
                    # Get all subjects that have data in both conditions and both periods
                    active_baseline = condition_data['active'][reference_period]
                    active_target = condition_data['active'][target_period]
                    sham_baseline = condition_data['sham'][reference_period]
                    sham_target = condition_data['sham'][target_period]
                    
                    # Find subjects with complete data
                    active_subjects = set(active_baseline['subject'].values) & set(active_target['subject'].values)
                    sham_subjects = set(sham_baseline['subject'].values) & set(sham_target['subject'].values)
                    common_subjects = active_subjects & sham_subjects
                    
                    if len(common_subjects) > 1:
                        for subject in common_subjects:
                            # Active session change
                            active_baseline_val = active_baseline[active_baseline['subject'] == subject][coeff].iloc[0]
                            active_target_val = active_target[active_target['subject'] == subject][coeff].iloc[0]
                            active_change = active_target_val - active_baseline_val
                            
                            # Sham session change
                            sham_baseline_val = sham_baseline[sham_baseline['subject'] == subject][coeff].iloc[0]
                            sham_target_val = sham_target[sham_target['subject'] == subject][coeff].iloc[0]
                            sham_change = sham_target_val - sham_baseline_val
                            
                            # Paired difference (active change - sham change)
                            paired_diff = active_change - sham_change
                            paired_changes.append(paired_diff)
                            
                            subject_info.append({
                                'subject': subject,
                                'active_change': active_change,
                                'sham_change': sham_change,
                                'paired_difference': paired_diff
                            })
                        
                        paired_changes = np.array(paired_changes)
                        active_changes = np.array([info['active_change'] for info in subject_info])
                        sham_changes = np.array([info['sham_change'] for info in subject_info])
                        
                        # Paired t-test (test if paired differences are different from zero)
                        t_stat_paired, p_val_paired = stats.ttest_1samp(paired_changes, 0)
                        
                        # One-sample t-tests for each condition vs zero
                        t_active_vs_zero, p_active_vs_zero = stats.ttest_1samp(active_changes, 0)
                        t_sham_vs_zero, p_sham_vs_zero = stats.ttest_1samp(sham_changes, 0)
                        
                        # Effect size for paired test (Cohen's d for paired samples)
                        cohens_d_paired = np.mean(paired_changes) / np.std(paired_changes, ddof=1) if np.std(paired_changes, ddof=1) > 0 else 0
                        
                        # Traditional unpaired comparison for reference
                        t_stat_unpaired, p_val_unpaired = stats.ttest_ind(active_changes, sham_changes)
                        pooled_std = np.sqrt(((len(active_changes) - 1) * np.var(active_changes, ddof=1) + 
                                             (len(sham_changes) - 1) * np.var(sham_changes, ddof=1)) / 
                                            (len(active_changes) + len(sham_changes) - 2))
                        cohens_d_unpaired = (np.mean(active_changes) - np.mean(sham_changes)) / pooled_std if pooled_std > 0 else 0
                        
                        comparison_results['coefficients'][coeff] = {
                            'n_paired_subjects': len(common_subjects),
                            
                            # Active condition statistics
                            'active_change_mean': float(np.mean(active_changes)),
                            'active_change_std': float(np.std(active_changes, ddof=1)),
                            'active_change_sem': float(np.std(active_changes, ddof=1) / np.sqrt(len(active_changes))),
                            
                            # Sham condition statistics  
                            'sham_change_mean': float(np.mean(sham_changes)),
                            'sham_change_std': float(np.std(sham_changes, ddof=1)),
                            'sham_change_sem': float(np.std(sham_changes, ddof=1) / np.sqrt(len(sham_changes))),
                            
                            # Paired differences
                            'paired_difference_mean': float(np.mean(paired_changes)),
                            'paired_difference_std': float(np.std(paired_changes, ddof=1)),
                            'paired_difference_sem': float(np.std(paired_changes, ddof=1) / np.sqrt(len(paired_changes))),
                            
                            # Paired t-test (PRIMARY TEST)
                            't_statistic_paired': float(t_stat_paired),
                            'p_value_paired': float(p_val_paired),
                            'cohens_d_paired': float(cohens_d_paired),
                            'degrees_of_freedom_paired': int(len(paired_changes) - 1),
                            
                            # One-sample tests (changes vs zero)
                            't_statistic_active_vs_zero': float(t_active_vs_zero),
                            'p_value_active_vs_zero': float(p_active_vs_zero),
                            't_statistic_sham_vs_zero': float(t_sham_vs_zero),
                            'p_value_sham_vs_zero': float(p_sham_vs_zero),
                            
                            # Unpaired comparison for reference
                            't_statistic_unpaired': float(t_stat_unpaired),
                            'p_value_unpaired': float(p_val_unpaired),
                            'cohens_d_unpaired': float(cohens_d_unpaired),
                            'degrees_of_freedom_unpaired': int(len(active_changes) + len(sham_changes) - 2),
                            
                            # Subject-level data (commented out to reduce file size)
                            # 'subject_data': subject_info
                        }
                    else:
                        comparison_results['coefficients'][coeff] = {'error': 'insufficient_paired_subjects'}
                        
                else:  # unpaired test (original approach)
                    # Calculate changes for each condition separately
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
                    
                    # Perform unpaired statistical test on changes
                    if len(active_changes) > 1 and len(sham_changes) > 1:
                        active_changes = np.array(active_changes)
                        sham_changes = np.array(sham_changes)
                        
                        # Unpaired t-test comparing active vs sham changes
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
                            
                            # Unpaired t-test (active vs sham changes)
                            't_statistic_unpaired': float(t_stat),
                            'p_value_unpaired': float(p_val),
                            'cohens_d_unpaired': float(cohens_d),
                            
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

def paired_baseline_corrected_roi_analysis(csv_file, output_file, test_type='paired', max_rois=None):
    """Perform baseline-corrected ROI analysis with paired or unpaired tests."""
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Data shape: {df.shape}")
    
    # Load labels
    labels = load_difumo_labels()
    
    # Get unique ROIs
    unique_rois = sorted(df['roi'].unique())
    if max_rois:
        unique_rois = unique_rois[:max_rois]
    
    print(f"Processing {len(unique_rois)} ROIs with {test_type} baseline-corrected analysis...")
    
    # Process each ROI
    roi_results = {}
    
    # For multiple comparison correction, collect p-values
    if test_type == 'paired':
        p_value_key = 'p_value_paired'
    else:
        p_value_key = 'p_value_unpaired'
    
    baseline_corrected_p_values = {coeff: {comparison: [] for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']} 
                                  for coeff in ['drift_0', 'drift_1', 'diffusion']}
    baseline_corrected_roi_mappings = {coeff: {comparison: [] for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']} 
                                      for coeff in ['drift_0', 'drift_1', 'diffusion']}
    
    for roi_idx in tqdm(unique_rois, desc="Processing ROIs"):
        roi_data = df[df['roi'] == roi_idx]
        
        # Get label
        roi_label = labels[roi_idx] if roi_idx < len(labels) else f"ROI_{roi_idx}"
        
        # Process statistics
        roi_stats = calculate_paired_baseline_corrected_changes(roi_data, test_type=test_type)
        
        roi_results[f"roi_{roi_idx:04d}"] = {
            'roi_index': int(roi_idx),
            'roi_label': roi_label,
            'analysis_results': roi_stats
        }
        
        # Collect p-values for multiple comparison correction
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_stats[comparison]['coefficients'][coeff]
                
                if p_value_key in coeff_data and not np.isnan(coeff_data[p_value_key]):
                    baseline_corrected_p_values[coeff][comparison].append(coeff_data[p_value_key])
                    baseline_corrected_roi_mappings[coeff][comparison].append(f"roi_{roi_idx:04d}")
    
    # Apply multiple comparison corrections
    print("Applying multiple comparison corrections...")
    
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
                correction_suffix = '_paired' if test_type == 'paired' else '_unpaired'
                for i, roi_key in enumerate(roi_keys):
                    coeff_results = roi_results[roi_key]['analysis_results'][comparison]['coefficients'][coeff]
                    coeff_results[f'p_value{correction_suffix}_bonferroni'] = float(p_bonf[i])
                    coeff_results[f'p_value{correction_suffix}_fdr'] = float(p_fdr[i])
    
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
            'analysis_type': 'baseline_corrected_only',
            'test_type': test_type,
            'baseline_corrected_comparisons': ['stimulation_minus_baseline', 'recovery_minus_baseline'],
            'statistical_tests': {
                'primary_test': f'{test_type}_t_test',
                'description': f'{"Paired t-test on within-subject differences (active_change - sham_change)" if test_type == "paired" else "Independent t-test comparing active vs sham change scores"}'
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
    significant_count = 0
    
    for roi_key, roi_data in roi_results.items():
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coeff in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results'][comparison]['coefficients'][coeff]
                if coeff_data.get(p_value_key, 1.0) < 0.05:
                    significant_count += 1
    
    total_tests = len(unique_rois) * 2 * 3  # ROIs * comparisons * coefficients
    
    print(f"\nQuick Summary:")
    print(f"  Test type: {test_type}")
    print(f"  Total tests: {total_tests}")
    print(f"  Significant ({test_type}, uncorrected p<0.05): {significant_count}")
    
    return output_data

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Paired baseline-corrected ROI-specific Fokker-Planck analysis')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with results')
    parser.add_argument('--output', type=str, default='paired_baseline_corrected_results.json',
                       help='Output JSON filename')
    parser.add_argument('--test-type', type=str, default='paired',
                       choices=['paired', 'unpaired'],
                       help='Type of statistical test (default: paired)')
    parser.add_argument('--max-rois', type=int, default=None,
                       help='Maximum number of ROIs to process (for testing)')
    
    args = parser.parse_args()
    
    # Run analysis
    output_path = f"/Users/jacekdmochowski/PROJECTS/fus_bold/code/{args.output}"
    results = paired_baseline_corrected_roi_analysis(
        args.input, 
        output_path, 
        test_type=args.test_type,
        max_rois=args.max_rois
    )

if __name__ == "__main__":
    main()