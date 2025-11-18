import numpy as np
import pandas as pd
import pickle
import sys
import os
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fokker_planck import fit_time_series_by_periods

def process_single_time_series(args):
    """Process a single time series. Designed for multiprocessing."""
    subject_idx, roi_idx, condition, time_series, dt = args
    
    try:
        results = fit_time_series_by_periods(time_series, dt=dt)
        
        output_rows = []
        for period, result in results.items():
            if 'error' not in result:
                row = {
                    'subject': subject_idx,
                    'roi': roi_idx,
                    'condition': condition,
                    'period': period,
                    'drift_0': result['drift_coeffs'][0],
                    'drift_1': result['drift_coeffs'][1] if len(result['drift_coeffs']) > 1 else 0,
                    'diffusion': result['diffusion_coeffs'][0],
                    'x_mean': result['x_mean'],
                    'x_std': result['x_std'],
                    'stationarity': result['diagnostics']['stationarity_test'],
                    'n_points': result['diagnostics']['n_points'],
                    'skewness': result['diagnostics']['skewness'],
                    'kurtosis': result['diagnostics']['kurtosis']
                }
                output_rows.append(row)
        
        return output_rows
        
    except Exception as e:
        # Return error information
        error_row = {
            'subject': subject_idx,
            'roi': roi_idx,
            'condition': condition,
            'period': 'error',
            'error': str(e)
        }
        return [error_row]

def batch_process_fokker_planck(n_subjects=None, n_rois=None, n_processes=4, 
                               output_file='fokker_planck_results_full.csv'):
    """
    Process all subjects and ROIs with Fokker-Planck fitting.
    
    Parameters:
    -----------
    n_subjects : int, optional
        Number of subjects to process (default: all 16)
    n_rois : int, optional
        Number of ROIs to process (default: all 1024)
    n_processes : int
        Number of parallel processes (default: 4)
    output_file : str
        Output CSV filename
    """
    
    # Load data
    print("Loading data...")
    data_path = '/Users/jacekdmochowski/PROJECTS/fus_bold/data/precomputed/difumo_time_series.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    n_subjects_total = len(data['active'])
    n_rois_total = data['active'][0].shape[1]
    
    if n_subjects is None:
        n_subjects = n_subjects_total
    if n_rois is None:
        n_rois = n_rois_total
        
    print(f"Processing {n_subjects} subjects, {n_rois} ROIs using {n_processes} processes")
    print(f"Total time series to process: {n_subjects * n_rois * 2} (active + sham)")
    
    # Prepare arguments for parallel processing
    args_list = []
    for subject_idx in range(n_subjects):
        for roi_idx in range(n_rois):
            for condition in ['active', 'sham']:
                time_series = data[condition][subject_idx][:, roi_idx]
                args_list.append((subject_idx, roi_idx, condition, time_series, 1.0))
    
    # Process in parallel
    print("Starting parallel processing...")
    
    if n_processes == 1:
        # Single-threaded processing
        results = []
        for args in tqdm(args_list, desc="Processing"):
            results.extend(process_single_time_series(args))
    else:
        # Multi-threaded processing
        with Pool(n_processes) as pool:
            results_nested = list(tqdm(
                pool.imap(process_single_time_series, args_list),
                total=len(args_list),
                desc="Processing"
            ))
        
        # Flatten results
        results = []
        for result_list in results_nested:
            results.extend(result_list)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Separate successful fits from errors
    df_success = df[df['period'] != 'error'].copy()
    df_errors = df[df['period'] == 'error'].copy()
    
    print(f"\nProcessing complete!")
    print(f"Successful fits: {len(df_success)}")
    print(f"Errors: {len(df_errors)}")
    
    if len(df_errors) > 0:
        print(f"Error rate: {len(df_errors) / len(args_list) * 100:.1f}%")
        
        # Show error summary
        print("\nError summary:")
        error_counts = df_errors['error'].value_counts()
        for error, count in error_counts.items():
            print(f"  {error}: {count} cases")
    
    # Save results
    output_path = f'/Users/jacekdmochowski/PROJECTS/fus_bold/code/{output_file}'
    df_success.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    if len(df_errors) > 0:
        error_path = f'/Users/jacekdmochowski/PROJECTS/fus_bold/code/{output_file.replace(".csv", "_errors.csv")}'
        df_errors.to_csv(error_path, index=False)
        print(f"Errors saved to: {error_path}")
    
    return df_success, df_errors

def analyze_results(df_results):
    """Analyze the batch processing results."""
    print("\n=== ANALYSIS OF BATCH RESULTS ===")
    
    # Basic statistics
    print(f"Total successful fits: {len(df_results)}")
    print(f"Subjects: {df_results['subject'].nunique()}")
    print(f"ROIs: {df_results['roi'].nunique()}")
    print(f"Conditions: {list(df_results['condition'].unique())}")
    print(f"Periods: {list(df_results['period'].unique())}")
    
    # Summary by condition and period
    print("\n=== SUMMARY BY CONDITION AND PERIOD ===")
    summary = df_results.groupby(['condition', 'period']).agg({
        'drift_0': ['mean', 'std', 'count'],
        'drift_1': ['mean', 'std'],
        'diffusion': ['mean', 'std']
    }).round(4)
    print(summary)
    
    # Effect of stimulation
    print("\n=== STIMULATION EFFECTS (Active vs Sham) ===")
    
    for period in ['baseline', 'stimulation', 'recovery']:
        active_data = df_results[(df_results['condition'] == 'active') & 
                                (df_results['period'] == period)]
        sham_data = df_results[(df_results['condition'] == 'sham') & 
                              (df_results['period'] == period)]
        
        if len(active_data) > 0 and len(sham_data) > 0:
            print(f"\n{period.upper()}:")
            
            # Statistical comparison
            from scipy import stats
            
            # Drift coefficient 0
            t_stat, p_val = stats.ttest_ind(active_data['drift_0'], sham_data['drift_0'])
            print(f"  Drift_0: t={t_stat:.3f}, p={p_val:.3f}")
            
            # Drift coefficient 1
            t_stat, p_val = stats.ttest_ind(active_data['drift_1'], sham_data['drift_1'])
            print(f"  Drift_1: t={t_stat:.3f}, p={p_val:.3f}")
            
            # Diffusion coefficient
            t_stat, p_val = stats.ttest_ind(active_data['diffusion'], sham_data['diffusion'])
            print(f"  Diffusion: t={t_stat:.3f}, p={p_val:.3f}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Batch process Fokker-Planck fitting')
    parser.add_argument('--subjects', type=int, default=None,
                       help='Number of subjects to process (default: all)')
    parser.add_argument('--rois', type=int, default=None,
                       help='Number of ROIs to process (default: all)')
    parser.add_argument('--processes', type=int, default=4,
                       help='Number of parallel processes (default: 4)')
    parser.add_argument('--output', type=str, default='fokker_planck_results_full.csv',
                       help='Output CSV filename')
    parser.add_argument('--analyze-only', type=str, default=None,
                       help='Skip processing and analyze existing results file')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load and analyze existing results
        df_results = pd.read_csv(args.analyze_only)
        analyze_results(df_results)
    else:
        # Run batch processing
        df_results, df_errors = batch_process_fokker_planck(
            n_subjects=args.subjects,
            n_rois=args.rois,
            n_processes=args.processes,
            output_file=args.output
        )
        
        # Analyze results
        if len(df_results) > 0:
            analyze_results(df_results)

if __name__ == "__main__":
    main()