#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import json
import os
from scipy import integrate
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def load_data():
    """Load time series data and Fokker-Planck results."""
    
    print("Loading time series data...")
    with open('../data/precomputed/difumo_time_series.pkl', 'rb') as f:
        difumo_ts = pickle.load(f)
    
    print("Loading Fokker-Planck results...")
    with open('fokker_planck_results_full.csv', 'r') as f:
        fp_results_df = pd.read_csv(f)
    
    return difumo_ts, fp_results_df

def extract_period_data(time_series, period):
    """Extract time series for specific period."""
    if period == 'baseline':
        return time_series[0:300]
    elif period == 'stimulation':
        return time_series[300:600]
    elif period == 'recovery':
        return time_series[600:900]
    else:
        raise ValueError(f"Unknown period: {period}")

def compute_steady_state_probability(x_range, drift_0, drift_1, diffusion):
    """
    Compute steady-state probability distribution from Fokker-Planck coefficients.
    
    p_ss(x) = C/D(x) * exp(∫ μ(x)/D(x) dx)
    where μ(x) = drift_0 + drift_1*x and D(x) = diffusion (constant)
    """
    
    # Avoid numerical issues
    if abs(diffusion) < 1e-8:
        diffusion = 1e-6
    
    # For constant diffusion, the integral is:
    # ∫ μ(x)/D dx = (drift_0*x + drift_1*x²/2)/diffusion
    integral_values = (drift_0 * x_range + drift_1 * x_range**2 / 2) / diffusion
    
    # Normalize for numerical stability
    integral_values = integral_values - np.max(integral_values)
    
    # Steady-state probability (unnormalized)
    p_unnorm = (1.0 / abs(diffusion)) * np.exp(integral_values)
    
    # Normalize
    dx = x_range[1] - x_range[0] if len(x_range) > 1 else 1.0
    normalization = np.trapezoid(p_unnorm, dx=dx)
    
    if normalization > 0 and np.isfinite(normalization):
        p_ss = p_unnorm / normalization
    else:
        # Fallback to uniform distribution
        p_ss = np.ones_like(x_range) / len(x_range)
    
    return p_ss

def compute_empirical_probability(time_series_data, x_range):
    """Compute empirical probability distribution using histogram."""
    
    # Use histogram approach for robustness
    counts, bin_edges = np.histogram(time_series_data, bins=len(x_range), 
                                   range=(x_range[0], x_range[-1]), density=False)
    
    # Convert to probability
    total_counts = np.sum(counts)
    if total_counts > 0:
        p_empirical = counts / total_counts
    else:
        p_empirical = np.ones_like(counts) / len(counts)
    
    # Handle edge case where we have fewer bins than x_range points
    if len(p_empirical) < len(x_range):
        # Interpolate or pad
        p_empirical = np.interp(x_range, 
                               (bin_edges[:-1] + bin_edges[1:]) / 2, 
                               p_empirical)
        p_empirical = p_empirical / np.sum(p_empirical)
    
    return p_empirical

def compute_potential_energy(x_range, drift_0, drift_1):
    """
    Compute potential energy φ(x) from drift coefficients.
    
    For constant diffusion: μ(x) = -dφ/dx
    So: φ(x) = -∫ μ(x) dx = -∫ (drift_0 + drift_1*x) dx = -drift_0*x - drift_1*x²/2
    """
    
    phi = -drift_0 * x_range - drift_1 * x_range**2 / 2
    
    return phi

def compute_free_energy(p_empirical, phi, kT=1.0):
    """
    Compute free energy: F = ∑_x p(x) φ(x) + kT ∑_x p(x) log p(x)
    """
    
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-12
    p_safe = np.maximum(p_empirical, epsilon)
    
    # Potential energy term
    potential_term = np.sum(p_empirical * phi)
    
    # Entropy term (negative entropy contribution)
    entropy_term = kT * np.sum(p_empirical * np.log(p_safe))
    
    # Total free energy
    free_energy = potential_term + entropy_term
    
    return free_energy, potential_term, entropy_term

def get_fp_coefficients(fp_results_df, roi_idx, condition, period):
    """Extract Fokker-Planck coefficients for specific ROI, condition, and period."""
    
    # Filter data for this ROI, condition, and period
    mask = (fp_results_df['roi'] == roi_idx) & \
           (fp_results_df['condition'] == condition) & \
           (fp_results_df['period'] == period)
    
    roi_data = fp_results_df[mask]
    
    if len(roi_data) == 0:
        # Return default values if no data found
        return 0.0, 0.0, 1.0
    
    # Get the first matching row (should be unique)
    row = roi_data.iloc[0]
    
    drift_0 = row.get('drift_0', 0.0)
    drift_1 = row.get('drift_1', 0.0) 
    diffusion = row.get('diffusion', 1.0)
    
    # Ensure diffusion is positive
    diffusion = abs(diffusion) if abs(diffusion) > 1e-8 else 1e-6
    
    return drift_0, drift_1, diffusion

def analyze_roi_free_energy(roi_idx, roi_label, time_series_data, fp_results_df):
    """Analyze free energy for a single ROI across all periods and conditions."""
    
    print(f"Analyzing ROI {roi_idx}: {roi_label[:50]}...")
    
    periods = ['baseline', 'stimulation', 'recovery']
    conditions = ['active', 'sham']
    
    results = {
        'roi_idx': roi_idx,
        'roi_label': roi_label,
        'periods': {}
    }
    
    for period in periods:
        results['periods'][period] = {}
        
        for condition in conditions:
            
            # Extract time series for this period across all subjects
            period_data = []
            for subj_idx in range(16):
                subj_ts = time_series_data[condition][subj_idx][:, roi_idx]
                period_ts = extract_period_data(subj_ts, period)
                period_data.extend(period_ts)
            
            period_data = np.array(period_data)
            
            # Get Fokker-Planck coefficients
            drift_0, drift_1, diffusion = get_fp_coefficients(fp_results_df, roi_idx, condition, period)
            
            # Define range for probability calculations
            data_min, data_max = np.percentile(period_data, [0.5, 99.5])  # Use robust range
            data_range = data_max - data_min
            if data_range < 1e-6:  # Handle constant signals
                data_range = 1.0
                
            x_range = np.linspace(data_min - 0.2*data_range, data_max + 0.2*data_range, 100)
            
            # Compute steady-state probability from Fokker-Planck
            p_steady_state = compute_steady_state_probability(x_range, drift_0, drift_1, diffusion)
            
            # Compute empirical probability
            p_empirical = compute_empirical_probability(period_data, x_range)
            
            # Compute potential energy
            phi = compute_potential_energy(x_range, drift_0, drift_1)
            
            # Compute free energy using empirical distribution
            free_energy, potential_term, entropy_term = compute_free_energy(p_empirical, phi)
            
            # Also compute theoretical free energy using steady-state distribution
            free_energy_theory, potential_theory, entropy_theory = compute_free_energy(p_steady_state, phi)
            
            results['periods'][period][condition] = {
                'fp_coefficients': {
                    'drift_0': drift_0,
                    'drift_1': drift_1,
                    'diffusion': diffusion
                },
                'free_energy_empirical': free_energy,
                'potential_energy_empirical': potential_term,
                'entropy_term_empirical': entropy_term,
                'free_energy_theoretical': free_energy_theory,
                'potential_energy_theoretical': potential_theory,
                'entropy_term_theoretical': entropy_theory,
                'data_statistics': {
                    'mean': np.mean(period_data),
                    'std': np.std(period_data),
                    'min': np.min(period_data),
                    'max': np.max(period_data),
                    'n_points': len(period_data)
                }
            }
    
    return results

def find_significant_rois(fp_results_df):
    """Find ROIs that have significant effects in the paired analysis."""
    
    print("Loading paired analysis results to identify significant ROIs...")
    with open('paired_full_results.json', 'r') as f:
        paired_results = json.load(f)
    
    significant_rois = []
    for roi_key, roi_data in paired_results['roi_results'].items():
        roi_idx = roi_data['roi_index']
        roi_label = roi_data['roi_label']
        
        # Check if ROI has any significant effects
        has_significant = False
        for comparison in ['stimulation_minus_baseline', 'recovery_minus_baseline']:
            for coefficient in ['drift_0', 'drift_1', 'diffusion']:
                coeff_data = roi_data['analysis_results'][comparison]['coefficients'][coefficient]
                p_val = coeff_data.get('p_value_paired', 1.0)
                if p_val < 0.05:
                    has_significant = True
                    break
            if has_significant:
                break
        
        if has_significant:
            significant_rois.append((roi_idx, roi_label))
    
    return significant_rois

def main():
    """Compute free energy for all ROIs with significant effects."""
    
    print("=== COMPUTING FREE ENERGY FROM FOKKER-PLANCK DYNAMICS (FULL ANALYSIS) ===")
    
    # Load data
    difumo_ts, fp_results_df = load_data()
    
    # Find ROIs with significant effects
    significant_rois = find_significant_rois(fp_results_df)
    print(f"Found {len(significant_rois)} ROIs with significant effects")
    
    # For computational efficiency, let's analyze first 20 ROIs
    # You can change this to analyze all ROIs
    n_rois_to_analyze = min(20, len(significant_rois))
    test_rois = significant_rois[:n_rois_to_analyze]
    print(f"Analyzing first {len(test_rois)} ROIs...")
    
    # Analyze free energy for selected ROIs
    all_results = []
    
    for i, (roi_idx, roi_label) in enumerate(test_rois):
        try:
            print(f"\\nProgress: {i+1}/{len(test_rois)}")
            results = analyze_roi_free_energy(roi_idx, roi_label, difumo_ts, fp_results_df)
            all_results.append(results)
        except Exception as e:
            print(f"Error analyzing ROI {roi_idx}: {e}")
    
    # Create summary DataFrame
    summary_data = []
    
    for results in all_results:
        roi_info = {
            'roi_idx': results['roi_idx'],
            'roi_label': results['roi_label']
        }
        
        for period in ['baseline', 'stimulation', 'recovery']:
            for condition in ['active', 'sham']:
                if period in results['periods'] and condition in results['periods'][period]:
                    period_data = results['periods'][period][condition]
                    
                    row = roi_info.copy()
                    row.update({
                        'period': period,
                        'condition': condition,
                        'free_energy_empirical': period_data['free_energy_empirical'],
                        'potential_energy_empirical': period_data['potential_energy_empirical'],
                        'entropy_term_empirical': period_data['entropy_term_empirical'],
                        'free_energy_theoretical': period_data['free_energy_theoretical'],
                        'potential_energy_theoretical': period_data['potential_energy_theoretical'],
                        'entropy_term_theoretical': period_data['entropy_term_theoretical'],
                        'drift_0': period_data['fp_coefficients']['drift_0'],
                        'drift_1': period_data['fp_coefficients']['drift_1'],
                        'diffusion': period_data['fp_coefficients']['diffusion'],
                        'mean_signal': period_data['data_statistics']['mean'],
                        'std_signal': period_data['data_statistics']['std'],
                        'min_signal': period_data['data_statistics']['min'],
                        'max_signal': period_data['data_statistics']['max'],
                        'n_points': period_data['data_statistics']['n_points']
                    })
                    
                    summary_data.append(row)
    
    # Save results
    df_summary = pd.DataFrame(summary_data)
    output_path = 'free_energy_analysis_full.csv'
    df_summary.to_csv(output_path, index=False)
    
    print(f"\\n=== FREE ENERGY ANALYSIS COMPLETE ===")
    print(f"Results saved to: {output_path}")
    print(f"Analyzed {len(all_results)} ROIs")
    print(f"Total measurements: {len(summary_data)}")
    
    # Compute change statistics
    if len(df_summary) > 0:
        print(f"\\nFree energy statistics (empirical):")
        print(f"Range: {df_summary['free_energy_empirical'].min():.4f} to {df_summary['free_energy_empirical'].max():.4f}")
        
        # Compare periods
        period_stats = df_summary.groupby(['period', 'condition'])['free_energy_empirical'].agg(['mean', 'std']).round(4)
        print(f"\\nMean free energy by period and condition:")
        print(period_stats)
        
        # Compute changes relative to baseline
        baseline_data = df_summary[df_summary['period'] == 'baseline'].set_index(['roi_idx', 'condition'])['free_energy_empirical']
        
        changes = []
        for period in ['stimulation', 'recovery']:
            period_data = df_summary[df_summary['period'] == period].set_index(['roi_idx', 'condition'])['free_energy_empirical']
            
            # Compute changes for matching ROIs and conditions
            common_idx = baseline_data.index.intersection(period_data.index)
            if len(common_idx) > 0:
                period_changes = period_data.loc[common_idx] - baseline_data.loc[common_idx]
                changes.extend([(period, 'active', roi_idx, change) for (roi_idx, condition), change in period_changes.items() if condition == 'active'])
                changes.extend([(period, 'sham', roi_idx, change) for (roi_idx, condition), change in period_changes.items() if condition == 'sham'])
        
        if changes:
            changes_df = pd.DataFrame(changes, columns=['period', 'condition', 'roi_idx', 'free_energy_change'])
            print(f"\\nFree energy changes from baseline:")
            change_stats = changes_df.groupby(['period', 'condition'])['free_energy_change'].agg(['mean', 'std', 'count']).round(4)
            print(change_stats)
    
    return df_summary

if __name__ == "__main__":
    results = main()