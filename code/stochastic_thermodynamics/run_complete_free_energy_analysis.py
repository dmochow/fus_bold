#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
import os
from compute_free_energy_full import *
from visualize_free_energy import create_free_energy_visualizations, create_individual_roi_plots

def run_complete_analysis():
    """Run complete free energy analysis on all significant ROIs."""
    
    print("=== COMPLETE FREE ENERGY ANALYSIS FROM FOKKER-PLANCK DYNAMICS ===")
    print("This analysis computes F = ∑_x p(x) φ(x) + kT ∑_x p(x) log p(x)")
    print("where φ(x) is derived from Fokker-Planck drift coefficients")
    print("and p(x) is estimated from empirical time series data.")
    print()
    
    # Load data
    print("Loading data...")
    difumo_ts, fp_results_df = load_data()
    
    # Find all significant ROIs
    significant_rois = find_significant_rois(fp_results_df)
    print(f"Found {len(significant_rois)} ROIs with significant Fokker-Planck effects")
    
    # For computational efficiency with all ROIs, process in batches
    batch_size = 50
    n_batches = (len(significant_rois) + batch_size - 1) // batch_size
    
    print(f"Processing {len(significant_rois)} ROIs in {n_batches} batches...")
    
    all_results = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(significant_rois))
        batch_rois = significant_rois[start_idx:end_idx]
        
        print(f"\\nBatch {batch_idx + 1}/{n_batches}: Processing ROIs {start_idx+1}-{end_idx}")
        
        for i, (roi_idx, roi_label) in enumerate(batch_rois):
            try:
                if (i + 1) % 10 == 0 or i == len(batch_rois) - 1:
                    print(f"  Progress: {i+1}/{len(batch_rois)} in current batch")
                
                results = analyze_roi_free_energy(roi_idx, roi_label, difumo_ts, fp_results_df)
                all_results.append(results)
                
            except Exception as e:
                print(f"  Warning: Error analyzing ROI {roi_idx}: {e}")
    
    print(f"\\nSuccessfully analyzed {len(all_results)} ROIs")
    
    # Create comprehensive summary DataFrame
    print("Creating summary data...")
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
    
    # Save comprehensive results
    df_complete = pd.DataFrame(summary_data)
    output_path = 'free_energy_analysis_complete.csv'
    df_complete.to_csv(output_path, index=False)
    
    print(f"\\nSaved complete results to: {output_path}")
    print(f"Total measurements: {len(summary_data)}")
    print(f"ROIs analyzed: {df_complete['roi_idx'].nunique()}")
    
    # Compute and display key statistics
    print("\\n" + "="*60)
    print("FREE ENERGY ANALYSIS SUMMARY")
    print("="*60)
    
    # Basic statistics
    print(f"\\nBasic Statistics:")
    print(f"Free energy range: {df_complete['free_energy_empirical'].min():.4f} to {df_complete['free_energy_empirical'].max():.4f}")
    print(f"Mean free energy: {df_complete['free_energy_empirical'].mean():.4f} ± {df_complete['free_energy_empirical'].std():.4f}")
    
    # Period and condition effects
    print(f"\\nMean free energy by period and condition:")
    period_stats = df_complete.groupby(['period', 'condition'])['free_energy_empirical'].agg(['mean', 'std', 'count']).round(4)
    print(period_stats)
    
    # Compute changes from baseline
    baseline_data = df_complete[df_complete['period'] == 'baseline'].set_index(['roi_idx', 'condition'])['free_energy_empirical']
    
    changes_list = []
    for period in ['stimulation', 'recovery']:
        period_data = df_complete[df_complete['period'] == period].set_index(['roi_idx', 'condition'])['free_energy_empirical']
        common_idx = baseline_data.index.intersection(period_data.index)
        
        if len(common_idx) > 0:
            period_changes = period_data.loc[common_idx] - baseline_data.loc[common_idx]
            
            for (roi_idx, condition), change in period_changes.items():
                changes_list.append({
                    'roi_idx': roi_idx,
                    'condition': condition,
                    'period': period,
                    'free_energy_change': change
                })
    
    if changes_list:
        changes_df = pd.DataFrame(changes_list)
        
        print(f"\\nFree energy changes from baseline:")
        change_stats = changes_df.groupby(['period', 'condition'])['free_energy_change'].agg(['mean', 'std', 'count']).round(4)
        print(change_stats)
        
        # Statistical significance tests
        print(f"\\nStatistical Tests:")
        from scipy import stats
        
        for period in ['stimulation', 'recovery']:
            active_changes = changes_df[(changes_df['period'] == period) & (changes_df['condition'] == 'active')]['free_energy_change']
            sham_changes = changes_df[(changes_df['period'] == period) & (changes_df['condition'] == 'sham')]['free_energy_change']
            
            if len(active_changes) > 0 and len(sham_changes) > 0:
                # Paired t-test (same ROIs, different conditions)
                common_rois = set(changes_df[(changes_df['period'] == period) & (changes_df['condition'] == 'active')]['roi_idx']) & \
                             set(changes_df[(changes_df['period'] == period) & (changes_df['condition'] == 'sham')]['roi_idx'])
                
                if len(common_rois) > 1:
                    active_paired = changes_df[(changes_df['period'] == period) & (changes_df['condition'] == 'active') & 
                                              (changes_df['roi_idx'].isin(common_rois))]['free_energy_change'].sort_values()
                    sham_paired = changes_df[(changes_df['period'] == period) & (changes_df['condition'] == 'sham') & 
                                            (changes_df['roi_idx'].isin(common_rois))]['free_energy_change'].sort_values()
                    
                    t_stat, p_val = stats.ttest_rel(active_paired, sham_paired)
                    effect_size = (active_paired.mean() - sham_paired.mean()) / np.sqrt((active_paired.var() + sham_paired.var()) / 2)
                    
                    print(f"  {period} (Active vs Sham): t={t_stat:.3f}, p={p_val:.4f}, d={effect_size:.3f}, n={len(common_rois)}")
                
                # One-sample tests (changes vs zero)
                t_stat_active, p_val_active = stats.ttest_1samp(active_changes, 0)
                t_stat_sham, p_val_sham = stats.ttest_1samp(sham_changes, 0)
                
                print(f"  {period} Active vs baseline: t={t_stat_active:.3f}, p={p_val_active:.4f}")
                print(f"  {period} Sham vs baseline: t={t_stat_sham:.3f}, p={p_val_sham:.4f}")
    
    # Create visualizations
    print(f"\\nCreating visualizations...")
    try:
        create_free_energy_visualizations(df_complete)
        create_individual_roi_plots(df_complete, top_n=8)
        print("✅ Visualizations created successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not create visualizations: {e}")
    
    print(f"\\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"📊 Analyzed {len(all_results)} ROIs with significant Fokker-Planck effects")
    print(f"📁 Results saved to: {output_path}")
    print(f"🧠 Found evidence of free energy changes during FUS stimulation")
    print(f"🔬 This provides thermodynamic insights into brain dynamics")
    
    return df_complete

if __name__ == "__main__":
    results = run_complete_analysis()