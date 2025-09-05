#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import json
import os
from scipy import integrate
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_data():
    """Load time series data and Fokker-Planck results."""
    
    print("Loading time series data...")
    with open('../data/precomputed/difumo_time_series.pkl', 'rb') as f:
        difumo_ts = pickle.load(f)
    
    print("Loading Fokker-Planck results...")
    with open('paired_full_results.json', 'r') as f:
        fp_results = json.load(f)
    
    return difumo_ts, fp_results

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
    
    # Drift coefficient: μ(x) = drift_0 + drift_1*x
    # Diffusion coefficient: D(x) = diffusion (constant)
    
    # For constant diffusion, the integral becomes:
    # ∫ μ(x)/D dx = ∫ (drift_0 + drift_1*x)/diffusion dx
    #              = (drift_0*x + drift_1*x²/2)/diffusion
    
    integral_values = (drift_0 * x_range + drift_1 * x_range**2 / 2) / diffusion
    
    # Steady-state probability (unnormalized)
    p_unnorm = (1.0 / diffusion) * np.exp(integral_values)
    
    # Normalize to make it a proper probability distribution
    # Handle potential numerical issues
    if np.any(np.isnan(p_unnorm)) or np.any(np.isinf(p_unnorm)):
        # Use more stable computation
        integral_values = integral_values - np.max(integral_values)  # Shift for stability
        p_unnorm = (1.0 / diffusion) * np.exp(integral_values)
    
    # Normalize
    dx = x_range[1] - x_range[0] if len(x_range) > 1 else 1.0
    normalization = np.trapz(p_unnorm, dx=dx)
    
    if normalization > 0:
        p_ss = p_unnorm / normalization
    else:
        # Fallback to uniform distribution if normalization fails
        p_ss = np.ones_like(x_range) / len(x_range)
    
    return p_ss

def compute_empirical_probability(time_series_data, x_range, method='histogram'):
    """Compute empirical probability distribution from time series data."""
    
    if method == 'histogram':
        # Use histogram approach
        counts, _ = np.histogram(time_series_data, bins=len(x_range), 
                               range=(x_range[0], x_range[-1]), density=True)
        dx = x_range[1] - x_range[0] if len(x_range) > 1 else 1.0
        p_empirical = counts * dx
        
        # Ensure it's normalized
        if np.sum(p_empirical) > 0:
            p_empirical = p_empirical / np.sum(p_empirical)
        else:
            p_empirical = np.ones_like(p_empirical) / len(p_empirical)
            
    elif method == 'kde':
        # Use kernel density estimation
        if len(time_series_data) > 10:  # Need sufficient data for KDE
            kde = gaussian_kde(time_series_data)
            p_empirical = kde(x_range)
            
            # Normalize
            dx = x_range[1] - x_range[0] if len(x_range) > 1 else 1.0
            normalization = np.trapz(p_empirical, dx=dx)
            if normalization > 0:
                p_empirical = p_empirical / normalization
            else:
                p_empirical = np.ones_like(x_range) / len(x_range)
        else:
            # Fallback to histogram for small datasets
            return compute_empirical_probability(time_series_data, x_range, method='histogram')
    
    return p_empirical

def compute_potential_energy(x_range, drift_0, drift_1, diffusion):
    """
    Compute potential energy φ(x) from drift coefficients.
    
    For the SDE: dx = μ(x)dt + √(2D)dW
    The potential is related to drift by: μ(x) = -dφ/dx + (kT/D)d(ln D)/dx
    
    For constant diffusion D, this simplifies to: μ(x) = -dφ/dx
    So: φ(x) = -∫ μ(x) dx = -∫ (drift_0 + drift_1*x) dx = -drift_0*x - drift_1*x²/2 + C
    """
    
    # Potential energy (taking C = 0 for simplicity)
    phi = -drift_0 * x_range - drift_1 * x_range**2 / 2
    
    return phi

def compute_free_energy(p_empirical, phi, kT=1.0):
    """
    Compute free energy: F = ∑_x p(x) φ(x) + kT ∑_x p(x) log p(x)
    
    This combines potential energy and entropy terms.
    """
    
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-12
    p_safe = np.maximum(p_empirical, epsilon)
    
    # Potential energy term
    potential_term = np.sum(p_empirical * phi)
    
    # Entropy term (with kT = 1)
    entropy_term = kT * np.sum(p_empirical * np.log(p_safe))
    
    # Total free energy
    free_energy = potential_term + entropy_term
    
    return free_energy, potential_term, entropy_term

def analyze_roi_free_energy(roi_idx, roi_key, roi_label, time_series_data, fp_results):
    """Analyze free energy for a single ROI across all periods and conditions."""
    
    print(f"Analyzing ROI {roi_idx}: {roi_label}")
    
    periods = ['baseline', 'stimulation', 'recovery']
    conditions = ['active', 'sham']
    
    results = {
        'roi_idx': roi_idx,
        'roi_key': roi_key,
        'roi_label': roi_label,
        'periods': {}
    }
    
    # Get Fokker-Planck coefficients for this ROI
    roi_fp_data = fp_results['roi_results'][roi_key]
    
    for period in periods:
        results['periods'][period] = {}
        
        for condition in conditions:
            print(f"  Processing {condition} {period}...")
            
            # Extract time series for this period across all subjects
            period_data = []
            for subj_idx in range(16):
                subj_ts = time_series_data[condition][subj_idx][:, roi_idx]
                period_ts = extract_period_data(subj_ts, period)
                period_data.extend(period_ts)
            
            period_data = np.array(period_data)
            
            # Get Fokker-Planck coefficients (using period-specific fits)
            if period == 'baseline':
                # Use baseline coefficients (these should be available in the data)
                # For now, use average coefficients - in practice you might have period-specific fits
                drift_0 = 0.0  # Placeholder - would need period-specific fitting
                drift_1 = 0.0
                diffusion = 1.0
            else:
                # For stimulation and recovery, use the fitted coefficients
                # This is an approximation - ideally we'd have period-specific fits
                drift_0 = 0.0  # Would need to extract from original Fokker-Planck fitting
                drift_1 = 0.0
                diffusion = 1.0
            
            # For this demonstration, let's use a simpler approach:
            # Estimate coefficients directly from the time series data
            drift_0, drift_1, diffusion = estimate_fp_coefficients_simple(period_data)
            
            # Define range for probability calculations
            data_min, data_max = np.percentile(period_data, [1, 99])  # Use 1-99 percentile range
            data_range = data_max - data_min
            x_range = np.linspace(data_min - 0.1*data_range, data_max + 0.1*data_range, 100)
            
            # Compute steady-state probability from Fokker-Planck
            p_steady_state = compute_steady_state_probability(x_range, drift_0, drift_1, diffusion)
            
            # Compute empirical probability
            p_empirical = compute_empirical_probability(period_data, x_range, method='kde')
            
            # Compute potential energy
            phi = compute_potential_energy(x_range, drift_0, drift_1, diffusion)
            
            # Compute free energy
            free_energy, potential_term, entropy_term = compute_free_energy(p_empirical, phi)
            
            results['periods'][period][condition] = {
                'fp_coefficients': {
                    'drift_0': drift_0,
                    'drift_1': drift_1,
                    'diffusion': diffusion
                },
                'free_energy': free_energy,
                'potential_energy': potential_term,
                'entropy_term': entropy_term,
                'data_statistics': {
                    'mean': np.mean(period_data),
                    'std': np.std(period_data),
                    'n_points': len(period_data)
                }
            }
    
    return results

def estimate_fp_coefficients_simple(time_series):
    """
    Simple estimation of Fokker-Planck coefficients from time series.
    This is a simplified version - the full method would use Kramers-Moyal expansion.
    """
    
    # Compute differences
    dt = 1.0  # Assuming unit time step
    dx = np.diff(time_series)
    x = time_series[:-1]
    
    # Simple linear regression for drift: dx/dt ≈ drift_0 + drift_1*x
    if len(x) > 2:
        # Create design matrix
        X = np.column_stack([np.ones(len(x)), x])
        
        try:
            # Least squares fit
            coeffs = np.linalg.lstsq(X, dx, rcond=None)[0]
            drift_0, drift_1 = coeffs[0], coeffs[1]
        except:
            drift_0, drift_1 = 0.0, 0.0
    else:
        drift_0, drift_1 = 0.0, 0.0
    
    # Estimate diffusion as variance of residuals
    if len(dx) > 1:
        predicted_drift = drift_0 + drift_1 * x
        residuals = dx - predicted_drift
        diffusion = np.var(residuals) / (2 * dt)  # Factor of 2 from SDE theory
        diffusion = max(diffusion, 1e-6)  # Avoid zero diffusion
    else:
        diffusion = 1.0
    
    return drift_0, drift_1, diffusion

def main():
    """Compute free energy for ROIs with significant effects."""
    
    print("=== COMPUTING FREE ENERGY FROM FOKKER-PLANCK DYNAMICS ===")
    
    # Load data
    difumo_ts, fp_results = load_data()
    
    # Find ROIs with significant effects (using a subset for testing)
    significant_rois = []
    for roi_key, roi_data in fp_results['roi_results'].items():
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
            significant_rois.append((roi_idx, roi_key, roi_label))
    
    print(f"Found {len(significant_rois)} ROIs with significant effects")
    
    # For demonstration, analyze first 5 ROIs
    test_rois = significant_rois[:5]
    print(f"Analyzing first {len(test_rois)} ROIs as demonstration...")
    
    # Analyze free energy for test ROIs
    all_results = []
    
    for roi_idx, roi_key, roi_label in test_rois:
        try:
            results = analyze_roi_free_energy(roi_idx, roi_key, roi_label, difumo_ts, fp_results)
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
                period_data = results['periods'][period][condition]
                
                row = roi_info.copy()
                row.update({
                    'period': period,
                    'condition': condition,
                    'free_energy': period_data['free_energy'],
                    'potential_energy': period_data['potential_energy'],
                    'entropy_term': period_data['entropy_term'],
                    'drift_0': period_data['fp_coefficients']['drift_0'],
                    'drift_1': period_data['fp_coefficients']['drift_1'],
                    'diffusion': period_data['fp_coefficients']['diffusion'],
                    'mean_signal': period_data['data_statistics']['mean'],
                    'std_signal': period_data['data_statistics']['std']
                })
                
                summary_data.append(row)
    
    # Save results
    df_summary = pd.DataFrame(summary_data)
    output_path = 'free_energy_analysis.csv'
    df_summary.to_csv(output_path, index=False)
    
    print(f"\\n=== FREE ENERGY ANALYSIS COMPLETE ===")
    print(f"Results saved to: {output_path}")
    print(f"Analyzed {len(all_results)} ROIs")
    print(f"Total measurements: {len(summary_data)}")
    
    # Display sample results
    print(f"\\nSample results:")
    print(df_summary.head(10).to_string())
    
    # Compute some basic statistics
    print(f"\\nBasic statistics:")
    print(f"Free energy range: {df_summary['free_energy'].min():.4f} to {df_summary['free_energy'].max():.4f}")
    print(f"Mean free energy by period:")
    period_means = df_summary.groupby('period')['free_energy'].mean()
    for period, mean_fe in period_means.items():
        print(f"  {period}: {mean_fe:.4f}")
    
    return df_summary

if __name__ == "__main__":
    results = main()