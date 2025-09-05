import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fokker_planck import FokkerPlanckFitter, fit_time_series_by_periods

def load_data():
    """Load the DiFuMo time series data."""
    data_path = '../data/precomputed/difumo_time_series.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def test_single_roi_single_subject():
    """Test fitting on a single ROI for a single subject."""
    print("Testing Fokker-Planck fitting on single ROI, single subject...")
    
    # Load data
    data = load_data()
    
    # Get first subject, first ROI, active condition
    subject_idx = 0
    roi_idx = 0
    time_series = data['active'][subject_idx][:, roi_idx]
    
    print(f"Time series shape: {time_series.shape}")
    print(f"Time series stats: mean={np.mean(time_series):.3f}, std={np.std(time_series):.3f}")
    
    # Fit by periods
    results = fit_time_series_by_periods(time_series, dt=1.0)
    
    # Print results
    for period, result in results.items():
        if 'error' in result:
            print(f"\n{period.upper()} PERIOD - ERROR: {result['error']}")
            continue
            
        print(f"\n{period.upper()} PERIOD:")
        print(f"  Drift coefficients: {result['drift_coeffs']}")
        print(f"  Diffusion coefficients: {result['diffusion_coeffs']}")
        print(f"  Data stats: mean={result['x_mean']:.3f}, std={result['x_std']:.3f}")
        print(f"  Diagnostics:")
        for key, value in result['diagnostics'].items():
            print(f"    {key}: {value:.4f}")
    
    return results

def test_multiple_rois():
    """Test fitting on multiple ROIs for single subject."""
    print("\nTesting multiple ROIs for single subject...")
    
    data = load_data()
    subject_idx = 0
    n_rois_test = 5  # Test first 5 ROIs
    
    results_summary = []
    
    for roi_idx in range(n_rois_test):
        print(f"\nProcessing ROI {roi_idx}...")
        
        for condition in ['active', 'sham']:
            time_series = data[condition][subject_idx][:, roi_idx]
            
            try:
                results = fit_time_series_by_periods(time_series, dt=1.0)
                
                for period, result in results.items():
                    if 'error' not in result:
                        summary = {
                            'subject': subject_idx,
                            'roi': roi_idx,
                            'condition': condition,
                            'period': period,
                            'drift_coeff_0': result['drift_coeffs'][0],
                            'drift_coeff_1': result['drift_coeffs'][1] if len(result['drift_coeffs']) > 1 else 0,
                            'diffusion_coeff': result['diffusion_coeffs'][0],
                            'x_mean': result['x_mean'],
                            'x_std': result['x_std'],
                            'stationarity': result['diagnostics']['stationarity_test']
                        }
                        results_summary.append(summary)
                        
            except Exception as e:
                print(f"  Error processing ROI {roi_idx}, condition {condition}: {e}")
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(results_summary)
    print(f"\nSummary of {len(df_summary)} successful fits:")
    print(df_summary.groupby(['condition', 'period']).agg({
        'drift_coeff_0': ['mean', 'std'],
        'drift_coeff_1': ['mean', 'std'], 
        'diffusion_coeff': ['mean', 'std']
    }).round(4))
    
    return df_summary

def visualize_example_fit():
    """Visualize an example fit with drift and diffusion functions."""
    print("\nCreating visualization of example fit...")
    
    data = load_data()
    subject_idx = 0
    roi_idx = 0
    time_series = data['active'][subject_idx][:, roi_idx]
    
    # Fit the model
    results = fit_time_series_by_periods(time_series, dt=1.0)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Fokker-Planck Fit Example: Subject {subject_idx}, ROI {roi_idx}', fontsize=14)
    
    # Plot time series with period boundaries
    ax = axes[0, 0]
    ax.plot(time_series, 'b-', alpha=0.7, linewidth=1)
    ax.axvline(x=300, color='r', linestyle='--', alpha=0.7, label='Stim start')
    ax.axvline(x=600, color='r', linestyle='--', alpha=0.7, label='Stim end')
    ax.set_xlabel('Time (TRs)')
    ax.set_ylabel('BOLD Signal')
    ax.set_title('Time Series with Period Boundaries')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot drift functions for each period
    ax = axes[0, 1]
    colors = ['blue', 'red', 'green']
    periods = ['baseline', 'stimulation', 'recovery']
    
    for i, period in enumerate(periods):
        if 'error' not in results[period]:
            fitter = results[period]['fitter']
            x_range, drift_vals = fitter.get_drift_function()
            ax.plot(x_range, drift_vals, color=colors[i], label=f'{period}', linewidth=2)
    
    ax.set_xlabel('BOLD Signal Value')
    ax.set_ylabel('Drift D1(x)')
    ax.set_title('Drift Functions by Period')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot diffusion functions
    ax = axes[1, 0]
    for i, period in enumerate(periods):
        if 'error' not in results[period]:
            fitter = results[period]['fitter']
            x_range, diff_vals = fitter.get_diffusion_function()
            ax.plot(x_range, diff_vals, color=colors[i], label=f'{period}', linewidth=2)
    
    ax.set_xlabel('BOLD Signal Value')
    ax.set_ylabel('Diffusion D2(x)')
    ax.set_title('Diffusion Functions by Period')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot coefficient comparison
    ax = axes[1, 1]
    drift_0_vals = []
    drift_1_vals = []
    diff_vals = []
    period_labels = []
    
    for period in periods:
        if 'error' not in results[period]:
            drift_0_vals.append(results[period]['drift_coeffs'][0])
            drift_1_vals.append(results[period]['drift_coeffs'][1] if len(results[period]['drift_coeffs']) > 1 else 0)
            diff_vals.append(results[period]['diffusion_coeffs'][0])
            period_labels.append(period)
    
    x_pos = np.arange(len(period_labels))
    width = 0.25
    
    ax.bar(x_pos - width, drift_0_vals, width, label='Drift (constant)', alpha=0.7)
    ax.bar(x_pos, drift_1_vals, width, label='Drift (linear)', alpha=0.7)
    ax.bar(x_pos + width, diff_vals, width, label='Diffusion', alpha=0.7)
    
    ax.set_xlabel('Period')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Coefficient Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(period_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../code/fokker_planck_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'fokker_planck_example.png'")

def main():
    """Run all tests."""
    print("=== Fokker-Planck Fitting Tests ===\n")
    
    try:
        # Test 1: Single ROI, single subject
        single_results = test_single_roi_single_subject()
        
        # Test 2: Multiple ROIs
        multi_results = test_multiple_rois()
        
        # Test 3: Visualization
        visualize_example_fit()
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()