import numpy as np
import pandas as pd
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fokker_planck import fit_time_series_by_periods

def analyze_coefficients():
    """Analyze Fokker-Planck coefficients for active vs sham conditions."""
    
    # Load data
    data_path = '../data/precomputed/difumo_time_series.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Analyzing Fokker-Planck coefficients...")
    print(f"Data: {len(data['active'])} subjects, {data['active'][0].shape[1]} ROIs")
    
    # Test on first few subjects and ROIs
    n_subjects_test = 3
    n_rois_test = 5
    
    results_list = []
    
    for subject_idx in range(n_subjects_test):
        print(f"\nProcessing subject {subject_idx}...")
        
        for roi_idx in range(n_rois_test):
            for condition in ['active', 'sham']:
                time_series = data[condition][subject_idx][:, roi_idx]
                
                try:
                    results = fit_time_series_by_periods(time_series, dt=1.0)
                    
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
                                'n_points': result['diagnostics']['n_points']
                            }
                            results_list.append(row)
                            
                except Exception as e:
                    print(f"  Error: Subject {subject_idx}, ROI {roi_idx}, {condition}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    print(f"\nSuccessfully analyzed {len(df)} time series segments")
    
    # Summary statistics
    print("\n=== SUMMARY BY CONDITION AND PERIOD ===")
    summary = df.groupby(['condition', 'period']).agg({
        'drift_0': ['mean', 'std', 'count'],
        'drift_1': ['mean', 'std'],
        'diffusion': ['mean', 'std']
    }).round(4)
    print(summary)
    
    # Compare active vs sham for each period
    print("\n=== ACTIVE vs SHAM DIFFERENCES (Mean ± SEM) ===")
    periods = ['baseline', 'stimulation', 'recovery']
    
    for period in periods:
        active_data = df[(df['condition'] == 'active') & (df['period'] == period)]
        sham_data = df[(df['condition'] == 'sham') & (df['period'] == period)]
        
        if len(active_data) > 0 and len(sham_data) > 0:
            print(f"\n{period.upper()}:")
            
            # Drift coefficient 0
            active_drift0 = active_data['drift_0'].values
            sham_drift0 = sham_data['drift_0'].values
            diff_drift0 = active_drift0.mean() - sham_drift0.mean()
            sem_diff0 = np.sqrt(active_drift0.var()/len(active_drift0) + sham_drift0.var()/len(sham_drift0))
            print(f"  Drift_0:    Active={active_drift0.mean():.4f}±{active_drift0.std()/np.sqrt(len(active_drift0)):.4f}, "
                  f"Sham={sham_drift0.mean():.4f}±{sham_drift0.std()/np.sqrt(len(sham_drift0)):.4f}, "
                  f"Diff={diff_drift0:.4f}±{sem_diff0:.4f}")
            
            # Drift coefficient 1
            active_drift1 = active_data['drift_1'].values
            sham_drift1 = sham_data['drift_1'].values
            diff_drift1 = active_drift1.mean() - sham_drift1.mean()
            sem_diff1 = np.sqrt(active_drift1.var()/len(active_drift1) + sham_drift1.var()/len(sham_drift1))
            print(f"  Drift_1:    Active={active_drift1.mean():.4f}±{active_drift1.std()/np.sqrt(len(active_drift1)):.4f}, "
                  f"Sham={sham_drift1.mean():.4f}±{sham_drift1.std()/np.sqrt(len(sham_drift1)):.4f}, "
                  f"Diff={diff_drift1:.4f}±{sem_diff1:.4f}")
            
            # Diffusion coefficient
            active_diff = active_data['diffusion'].values
            sham_diff = sham_data['diffusion'].values
            diff_diffusion = active_diff.mean() - sham_diff.mean()
            sem_diffusion = np.sqrt(active_diff.var()/len(active_diff) + sham_diff.var()/len(sham_diff))
            print(f"  Diffusion:  Active={active_diff.mean():.4f}±{active_diff.std()/np.sqrt(len(active_diff)):.4f}, "
                  f"Sham={sham_diff.mean():.4f}±{sham_diff.std()/np.sqrt(len(sham_diff)):.4f}, "
                  f"Diff={diff_diffusion:.4f}±{sem_diffusion:.4f}")
    
    # Save results
    df.to_csv('../code/fokker_planck_results.csv', index=False)
    print(f"\nResults saved to 'fokker_planck_results.csv'")
    
    return df

if __name__ == "__main__":
    df_results = analyze_coefficients()