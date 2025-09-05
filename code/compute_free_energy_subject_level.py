#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
import json
import os
from scipy import stats
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
        # Interpolate to match x_range length
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        p_empirical = np.interp(x_range, bin_centers, p_empirical)
        p_empirical = p_empirical / np.sum(p_empirical)
    
    return p_empirical


def compute_potential_energy(x_range, drift_0, drift_1, diffusion):
    """
    Compute potential energy φ(x) from drift coefficients and diffusion.

    For constant diffusion: μ(x) = drift_0 + drift_1 * x
    φ(x) = -∫ μ(x)/D dx = -1/D * (drift_0 * x + drift_1 * x^2 / 2)
    """
    if abs(diffusion) < 1e-8:
        diffusion = 1e-6  # safeguard

    phi = -(drift_0 * x_range + 0.5 * drift_1 * x_range ** 2) / diffusion
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

def get_subject_fp_coefficients(fp_results_df, subject_idx, roi_idx, condition, period):
    """Extract Fokker-Planck coefficients for specific subject, ROI, condition, and period."""
    
    # Filter data for this subject, ROI, condition, and period
    mask = (fp_results_df['subject'] == subject_idx) & \
           (fp_results_df['roi'] == roi_idx) & \
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

def get_roi_label(roi_idx):
    """Get ROI label from the paired results."""
    try:
        with open('paired_full_results.json', 'r') as f:
            paired_results = json.load(f)
        
        roi_key = f'roi_{roi_idx:04d}'
        if roi_key in paired_results['roi_results']:
            return paired_results['roi_results'][roi_key]['roi_label']
        else:
            return f'ROI_{roi_idx:04d}'
    except:
        return f'ROI_{roi_idx:04d}'

def analyze_subject_roi_free_energy(subject_idx, roi_idx, time_series_data, fp_results_df):
    """Analyze free energy for a single subject and ROI across all periods and conditions."""
    
    periods = ['baseline', 'stimulation', 'recovery']
    conditions = ['active', 'sham']
    
    results = {
        'subject_idx': subject_idx,
        'roi_idx': roi_idx,
        'periods': {}
    }
    
    for condition in conditions:
        for period in periods:
            
            # Extract time series for this subject, ROI, period, and condition
            subj_ts = time_series_data[condition][subject_idx][:, roi_idx]
            period_ts = extract_period_data(subj_ts, period)
            
            # Get subject-specific Fokker-Planck coefficients
            drift_0, drift_1, diffusion = get_subject_fp_coefficients(
                fp_results_df, subject_idx, roi_idx, condition, period)
            
            # Define range for probability calculations based on this subject's data
            data_min, data_max = np.percentile(period_ts, [0.5, 99.5])
            data_range = data_max - data_min
            if data_range < 1e-6:  # Handle constant signals
                data_range = 1.0
                
            x_range = np.linspace(data_min - 0.2*data_range, data_max + 0.2*data_range, 100)
            
            # Compute steady-state probability from Fokker-Planck
            p_steady_state = compute_steady_state_probability(x_range, drift_0, drift_1, diffusion)
            
            # Compute empirical probability
            p_empirical = compute_empirical_probability(period_ts, x_range)
            
            # Compute potential energy
            phi = compute_potential_energy(x_range, drift_0, drift_1, diffusion)
            
            # Compute free energy using empirical distribution
            free_energy, potential_term, entropy_term = compute_free_energy(p_empirical, phi)
            
            # Also compute theoretical free energy using steady-state distribution
            free_energy_theory, potential_theory, entropy_theory = compute_free_energy(p_steady_state, phi)
            
            # Store results
            key = f'{condition}_{period}'
            results['periods'][key] = {
                'condition': condition,
                'period': period,
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
                    'mean': np.mean(period_ts),
                    'std': np.std(period_ts),
                    'min': np.min(period_ts),
                    'max': np.max(period_ts),
                    'n_points': len(period_ts)
                }
            }
    
    return results

def compute_free_energy_changes(results):
    """Compute free energy changes from baseline for each subject and condition."""
    
    changes = {}
    
    for condition in ['active', 'sham']:
        baseline_key = f'{condition}_baseline'
        
        if baseline_key in results['periods']:
            baseline_fe = results['periods'][baseline_key]['free_energy_empirical']
            
            for period in ['stimulation', 'recovery']:
                period_key = f'{condition}_{period}'
                
                if period_key in results['periods']:
                    period_fe = results['periods'][period_key]['free_energy_empirical']
                    change = period_fe - baseline_fe
                    
                    changes[f'{condition}_{period}_minus_baseline'] = {
                        'condition': condition,
                        'comparison': f'{period}_minus_baseline',
                        'free_energy_change': change,
                        'baseline_fe': baseline_fe,
                        'period_fe': period_fe
                    }
    
    return changes

def analyze_roi_statistics(roi_data):
    """Perform statistical analysis for a single ROI across all subjects."""
    
    # Extract changes for statistical testing
    active_stim_changes = []
    sham_stim_changes = []
    active_rec_changes = []
    sham_rec_changes = []
    
    for subject_results in roi_data:
        changes = compute_free_energy_changes(subject_results)
        
        if 'active_stimulation_minus_baseline' in changes:
            active_stim_changes.append(changes['active_stimulation_minus_baseline']['free_energy_change'])
        
        if 'sham_stimulation_minus_baseline' in changes:
            sham_stim_changes.append(changes['sham_stimulation_minus_baseline']['free_energy_change'])
            
        if 'active_recovery_minus_baseline' in changes:
            active_rec_changes.append(changes['active_recovery_minus_baseline']['free_energy_change'])
            
        if 'sham_recovery_minus_baseline' in changes:
            sham_rec_changes.append(changes['sham_recovery_minus_baseline']['free_energy_change'])
    
    # Convert to numpy arrays
    active_stim_changes = np.array(active_stim_changes)
    sham_stim_changes = np.array(sham_stim_changes)
    active_rec_changes = np.array(active_rec_changes)
    sham_rec_changes = np.array(sham_rec_changes)
    
    statistics = {}
    
    # Paired t-tests comparing active vs sham changes
    if len(active_stim_changes) > 0 and len(sham_stim_changes) > 0 and len(active_stim_changes) == len(sham_stim_changes):
        t_stat_stim, p_val_stim = stats.ttest_rel(active_stim_changes, sham_stim_changes)
        effect_size_stim = (np.mean(active_stim_changes) - np.mean(sham_stim_changes)) / \
                          np.sqrt((np.var(active_stim_changes, ddof=1) + np.var(sham_stim_changes, ddof=1)) / 2)
        
        statistics['stimulation_vs_baseline'] = {
            'active_mean_change': np.mean(active_stim_changes),
            'active_std_change': np.std(active_stim_changes, ddof=1),
            'sham_mean_change': np.mean(sham_stim_changes),
            'sham_std_change': np.std(sham_stim_changes, ddof=1),
            't_statistic': t_stat_stim,
            'p_value': p_val_stim,
            'cohens_d': effect_size_stim,
            'n_subjects': len(active_stim_changes)
        }
    
    if len(active_rec_changes) > 0 and len(sham_rec_changes) > 0 and len(active_rec_changes) == len(sham_rec_changes):
        t_stat_rec, p_val_rec = stats.ttest_rel(active_rec_changes, sham_rec_changes)
        effect_size_rec = (np.mean(active_rec_changes) - np.mean(sham_rec_changes)) / \
                         np.sqrt((np.var(active_rec_changes, ddof=1) + np.var(sham_rec_changes, ddof=1)) / 2)
        
        statistics['recovery_vs_baseline'] = {
            'active_mean_change': np.mean(active_rec_changes),
            'active_std_change': np.std(active_rec_changes, ddof=1),
            'sham_mean_change': np.mean(sham_rec_changes),
            'sham_std_change': np.std(sham_rec_changes, ddof=1),
            't_statistic': t_stat_rec,
            'p_value': p_val_rec,
            'cohens_d': effect_size_rec,
            'n_subjects': len(active_rec_changes)
        }
    
    return statistics

def main():
    """Compute free energy at the subject level for all ROIs."""
    
    print("=== SUBJECT-LEVEL FREE ENERGY ANALYSIS FOR ALL ROIS ===")
    print("Computing free energy for each individual subject using their own Fokker-Planck coefficients")
    print("Statistical testing: paired t-tests of (Active - Baseline) vs (Sham - Baseline) changes")
    print()
    
    # Load data
    difumo_ts, fp_results_df = load_data()
    
    # Get all ROI indices (0 to 1023)
    all_rois = list(range(1024))
    print(f"Processing all {len(all_rois)} ROIs at subject level...")
    
    # Process in smaller batches for memory efficiency
    batch_size = 50
    n_batches = (len(all_rois) + batch_size - 1) // batch_size
    
    print(f"Processing in {n_batches} batches of {batch_size} ROIs each...")
    print("Each ROI will be analyzed for all 16 subjects individually...")
    
    # Store all results
    all_subject_data = []
    all_roi_statistics = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_rois))
        batch_rois = all_rois[start_idx:end_idx]
        
        print(f"\\nBatch {batch_idx + 1}/{n_batches}: Processing ROIs {start_idx}-{end_idx-1}")
        
        for roi_idx in batch_rois:
            roi_label = get_roi_label(roi_idx)
            
            # Analyze all subjects for this ROI
            roi_subject_results = []
            
            for subject_idx in range(16):
                try:
                    subject_results = analyze_subject_roi_free_energy(
                        subject_idx, roi_idx, difumo_ts, fp_results_df)
                    roi_subject_results.append(subject_results)
                    
                    # Add to overall subject data
                    subject_results['roi_label'] = roi_label
                    all_subject_data.append(subject_results)
                    
                except Exception as e:
                    print(f"  Warning: Error analyzing subject {subject_idx}, ROI {roi_idx}: {e}")
            
            # Perform statistical analysis for this ROI
            if len(roi_subject_results) > 0:
                try:
                    roi_stats = analyze_roi_statistics(roi_subject_results)
                    roi_stats['roi_idx'] = roi_idx
                    roi_stats['roi_label'] = roi_label
                    roi_stats['n_subjects_analyzed'] = len(roi_subject_results)
                    all_roi_statistics.append(roi_stats)
                except Exception as e:
                    print(f"  Warning: Error computing statistics for ROI {roi_idx}: {e}")
        
        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            print(f"  Progress: {len(all_subject_data)} subject-ROI combinations completed")
    
    print(f"\\nSuccessfully analyzed {len(all_subject_data)} subject-ROI combinations")
    print(f"Computed statistics for {len(all_roi_statistics)} ROIs")
    
    # Create subject-level DataFrame
    print("Creating subject-level summary data...")
    subject_summary_data = []
    
    for subject_results in all_subject_data:
        subject_info = {
            'subject_idx': subject_results['subject_idx'],
            'roi_idx': subject_results['roi_idx'],
            'roi_label': subject_results['roi_label']
        }
        
        # Add changes from baseline
        changes = compute_free_energy_changes(subject_results)
        
        for change_key, change_data in changes.items():
            row = subject_info.copy()
            row.update({
                'condition': change_data['condition'],
                'comparison': change_data['comparison'],
                'free_energy_change': change_data['free_energy_change'],
                'baseline_free_energy': change_data['baseline_fe'],
                'period_free_energy': change_data['period_fe']
            })
            subject_summary_data.append(row)
        
        # Also add absolute values for each period/condition
        for period_key, period_data in subject_results['periods'].items():
            row = subject_info.copy()
            row.update({
                'condition': period_data['condition'],
                'comparison': period_data['period'],  # This is the absolute period, not a comparison
                'free_energy_change': np.nan,  # Not applicable for absolute values
                'baseline_free_energy': np.nan,
                'period_free_energy': period_data['free_energy_empirical']
            })
            # Add FP coefficients
            row.update({
                'drift_0': period_data['fp_coefficients']['drift_0'],
                'drift_1': period_data['fp_coefficients']['drift_1'],
                'diffusion': period_data['fp_coefficients']['diffusion']
            })
            subject_summary_data.append(row)
    
    # Create ROI statistics DataFrame
    roi_stats_data = []
    for roi_stats in all_roi_statistics:
        roi_info = {
            'roi_idx': roi_stats['roi_idx'],
            'roi_label': roi_stats['roi_label'],
            'n_subjects_analyzed': roi_stats['n_subjects_analyzed']
        }
        
        for comparison in ['stimulation_vs_baseline', 'recovery_vs_baseline']:
            if comparison in roi_stats:
                row = roi_info.copy()
                row.update({
                    'comparison': comparison,
                    'active_mean_change': roi_stats[comparison]['active_mean_change'],
                    'active_std_change': roi_stats[comparison]['active_std_change'],
                    'sham_mean_change': roi_stats[comparison]['sham_mean_change'],
                    'sham_std_change': roi_stats[comparison]['sham_std_change'],
                    't_statistic': roi_stats[comparison]['t_statistic'],
                    'p_value': roi_stats[comparison]['p_value'],
                    'cohens_d': roi_stats[comparison]['cohens_d'],
                    'n_subjects': roi_stats[comparison]['n_subjects']
                })
                roi_stats_data.append(row)
    
    # Save results
    subject_df = pd.DataFrame(subject_summary_data)
    roi_stats_df = pd.DataFrame(roi_stats_data)
    
    subject_output_path = 'free_energy_subject_level.csv'
    roi_stats_output_path = 'free_energy_roi_statistics.csv'
    
    subject_df.to_csv(subject_output_path, index=False)
    roi_stats_df.to_csv(roi_stats_output_path, index=False)
    
    print(f"\\n=== SUBJECT-LEVEL FREE ENERGY ANALYSIS COMPLETE ===")
    print(f"Subject-level data saved to: {subject_output_path}")
    print(f"ROI statistics saved to: {roi_stats_output_path}")
    print(f"Total subject-level measurements: {len(subject_summary_data)}")
    print(f"ROIs with statistical results: {len(roi_stats_data)}")
    
    # Display key findings
    if len(roi_stats_df) > 0:
        print(f"\\nKey Statistical Findings:")
        
        # Count significant effects
        significant_stim = roi_stats_df[
            (roi_stats_df['comparison'] == 'stimulation_vs_baseline') & 
            (roi_stats_df['p_value'] < 0.05)
        ]
        
        significant_rec = roi_stats_df[
            (roi_stats_df['comparison'] == 'recovery_vs_baseline') & 
            (roi_stats_df['p_value'] < 0.05)
        ]
        
        print(f"Significant ROIs (p < 0.05):")
        print(f"  Stimulation vs Baseline: {len(significant_stim)} / {len(roi_stats_df[roi_stats_df['comparison'] == 'stimulation_vs_baseline'])}")
        print(f"  Recovery vs Baseline: {len(significant_rec)} / {len(roi_stats_df[roi_stats_df['comparison'] == 'recovery_vs_baseline'])}")
        
        if len(significant_stim) > 0:
            print(f"\\nStimulation effects:")
            print(f"  Mean effect size (Cohen's d): {significant_stim['cohens_d'].mean():.3f}")
            print(f"  Effect size range: {significant_stim['cohens_d'].min():.3f} to {significant_stim['cohens_d'].max():.3f}")
        
        if len(significant_rec) > 0:
            print(f"\\nRecovery effects:")
            print(f"  Mean effect size (Cohen's d): {significant_rec['cohens_d'].mean():.3f}")
            print(f"  Effect size range: {significant_rec['cohens_d'].min():.3f} to {significant_rec['cohens_d'].max():.3f}")
    
    return subject_df, roi_stats_df

if __name__ == "__main__":
    subject_results, roi_statistics = main()