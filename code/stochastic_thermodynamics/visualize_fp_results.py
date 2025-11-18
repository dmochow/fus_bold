import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fokker_planck import fit_time_series_by_periods

def visualize_fokker_planck_results():
    """Create visualization of Fokker-Planck fitting results."""
    
    # Load data
    data_path = '../data/precomputed/difumo_time_series.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Analyze first subject, first ROI for both conditions
    subject_idx = 0
    roi_idx = 0
    
    results = {}
    for condition in ['active', 'sham']:
        time_series = data[condition][subject_idx][:, roi_idx]
        results[condition] = fit_time_series_by_periods(time_series, dt=1.0)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Fokker-Planck Analysis: Subject {subject_idx}, ROI {roi_idx}', fontsize=16)
    
    conditions = ['active', 'sham']
    periods = ['baseline', 'stimulation', 'recovery']
    colors = ['blue', 'red', 'green']
    
    for col, condition in enumerate(conditions):
        time_series = data[condition][subject_idx][:, roi_idx]
        
        # Plot 1: Time series with period boundaries
        ax = axes[0, col]
        ax.plot(time_series, 'k-', alpha=0.7, linewidth=1)
        ax.axvline(x=300, color='r', linestyle='--', alpha=0.7, label='Stim start')
        ax.axvline(x=600, color='r', linestyle='--', alpha=0.7, label='Stim end')
        ax.set_xlabel('Time (TRs)')
        ax.set_ylabel('BOLD Signal')
        ax.set_title(f'{condition.capitalize()} - Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Drift functions
        ax = axes[1, col]
        for i, period in enumerate(periods):
            if 'error' not in results[condition][period]:
                fitter = results[condition][period]['fitter']
                x_range, drift_vals = fitter.get_drift_function()
                ax.plot(x_range, drift_vals, color=colors[i], label=f'{period}', linewidth=2)
        
        ax.set_xlabel('BOLD Signal Value')
        ax.set_ylabel('Drift D1(x)')
        ax.set_title(f'{condition.capitalize()} - Drift Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Diffusion functions
        ax = axes[2, col]
        for i, period in enumerate(periods):
            if 'error' not in results[condition][period]:
                fitter = results[condition][period]['fitter']
                x_range, diff_vals = fitter.get_diffusion_function()
                ax.plot(x_range, diff_vals, color=colors[i], label=f'{period}', linewidth=2)
        
        ax.set_xlabel('BOLD Signal Value')
        ax.set_ylabel('Diffusion D2(x)')
        ax.set_title(f'{condition.capitalize()} - Diffusion Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../code/fokker_planck_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print coefficient comparison
    print("\n=== COEFFICIENT COMPARISON ===")
    print(f"{'Condition':<10} {'Period':<12} {'Drift_0':<10} {'Drift_1':<10} {'Diffusion':<10}")
    print("-" * 60)
    
    for condition in conditions:
        for period in periods:
            if 'error' not in results[condition][period]:
                res = results[condition][period]
                drift_0 = res['drift_coeffs'][0]
                drift_1 = res['drift_coeffs'][1] if len(res['drift_coeffs']) > 1 else 0
                diffusion = res['diffusion_coeffs'][0]
                print(f"{condition:<10} {period:<12} {drift_0:<10.4f} {drift_1:<10.4f} {diffusion:<10.4f}")
    
    # Calculate differences between active and sham
    print("\n=== ACTIVE vs SHAM DIFFERENCES ===")
    print(f"{'Period':<12} {'Δ Drift_0':<12} {'Δ Drift_1':<12} {'Δ Diffusion':<12}")
    print("-" * 50)
    
    for period in periods:
        if ('error' not in results['active'][period] and 
            'error' not in results['sham'][period]):
            
            active_res = results['active'][period]
            sham_res = results['sham'][period]
            
            diff_drift_0 = active_res['drift_coeffs'][0] - sham_res['drift_coeffs'][0]
            diff_drift_1 = (active_res['drift_coeffs'][1] - sham_res['drift_coeffs'][1] 
                           if len(active_res['drift_coeffs']) > 1 else 0)
            diff_diffusion = active_res['diffusion_coeffs'][0] - sham_res['diffusion_coeffs'][0]
            
            print(f"{period:<12} {diff_drift_0:<12.4f} {diff_drift_1:<12.4f} {diff_diffusion:<12.4f}")

if __name__ == "__main__":
    visualize_fokker_planck_results()