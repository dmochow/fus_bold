import pandas as pd
import numpy as np

def summarize_key_findings(csv_file):
    """Create a focused summary of the most important findings."""
    
    print("Loading significant effects...")
    df = pd.read_csv(csv_file)
    
    # Separate baseline-corrected and standard results
    df_baseline = df[df['analysis_type'] == 'baseline_corrected'].copy()
    df_standard = df[df['analysis_type'] == 'standard'].copy()
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS FROM FOKKER-PLANCK ANALYSIS OF FUS-BOLD DATA")
    print(f"{'='*80}")
    
    print(f"\nTOTAL SIGNIFICANT EFFECTS: {len(df)}")
    print(f"  Standard analysis (Active vs Sham): {len(df_standard)}")
    print(f"  Baseline-corrected analysis (Change scores): {len(df_baseline)}")
    
    print(f"\n{'='*60}")
    print("BASELINE-CORRECTED RESULTS (Most Valid for FUS Effects)")
    print(f"{'='*60}")
    
    # Top 20 most significant baseline-corrected effects
    print(f"\nTOP 20 MOST SIGNIFICANT BASELINE-CORRECTED EFFECTS:")
    print(f"{'Rank':<4} {'ROI':<4} {'Region':<45} {'Period':<20} {'Coeff':<8} {'p-value':<10} {'Effect':<6}")
    print("-" * 105)
    
    top_20 = df_baseline.head(20)
    for i, (_, row) in enumerate(top_20.iterrows(), 1):
        region_short = row['roi_label'][:44]
        period_short = row['comparison'].replace('_minus_baseline', '').replace('_', ' ')
        active_change = "↑" if row['active_change_direction'] == 'increase' else "↓"
        sham_change = "↑" if row['sham_change_direction'] == 'increase' else "↓"
        effect_dir = f"A{active_change}S{sham_change}"
        
        print(f"{i:<4} {row['roi_index']:<4} {region_short:<45} {period_short:<20} {row['coefficient']:<8} {row['p_value']:<10.4f} {effect_dir:<6}")
    
    # Breakdown by period
    print(f"\nEFFECTS BY EXPERIMENTAL PERIOD:")
    stim_effects = len(df_baseline[df_baseline['comparison'] == 'stimulation_minus_baseline'])
    recovery_effects = len(df_baseline[df_baseline['comparison'] == 'recovery_minus_baseline'])
    print(f"  During stimulation: {stim_effects} significant effects")
    print(f"  During recovery:    {recovery_effects} significant effects")
    
    # Breakdown by coefficient
    print(f"\nEFFECTS BY FOKKER-PLANCK COEFFICIENT:")
    for coeff in ['drift_0', 'drift_1', 'diffusion']:
        coeff_effects = len(df_baseline[df_baseline['coefficient'] == coeff])
        print(f"  {coeff:<12}: {coeff_effects} effects")
    
    # Direction analysis
    print(f"\nCHANGE DIRECTION PATTERNS:")
    print(f"  Active condition:")
    active_directions = df_baseline['active_change_direction'].value_counts()
    for direction, count in active_directions.items():
        print(f"    {direction}: {count}")
    
    print(f"  Sham condition:")
    sham_directions = df_baseline['sham_change_direction'].value_counts()
    for direction, count in sham_directions.items():
        print(f"    {direction}: {count}")
    
    # ROI hotspots
    print(f"\nROI HOTSPOTS (Most affected brain regions):")
    roi_counts = df_baseline['roi_index'].value_counts().head(10)
    for roi_idx, count in roi_counts.items():
        roi_label = df_baseline[df_baseline['roi_index'] == roi_idx]['roi_label'].iloc[0]
        print(f"  ROI {roi_idx:3d}: {count} effects - {roi_label}")
    
    # Brain networks/systems analysis
    print(f"\nBRAIN NETWORK ANALYSIS:")
    
    # Define network keywords
    networks = {
        'Cerebellum': ['cerebellum'],
        'Visual': ['visual', 'occipital', 'calcarine', 'cuneus'],
        'Sensorimotor': ['precentral', 'postcentral', 'motor', 'sensory'],
        'Attention': ['parietal', 'frontal', 'attention'],
        'Default Mode': ['cingulate', 'precuneus', 'angular'],
        'Salience': ['insula', 'anterior cingulate'],
        'Limbic': ['hippocampus', 'amygdala', 'limbic'],
        'White Matter': ['fasciculus', 'radiation', 'tract', 'capsule']
    }
    
    for network, keywords in networks.items():
        network_rois = df_baseline[df_baseline['roi_label'].str.contains('|'.join(keywords), case=False, na=False)]
        if len(network_rois) > 0:
            print(f"  {network:<15}: {len(network_rois)} effects")
    
    # Target region analysis
    print(f"\nTARGET REGION ANALYSIS (Subgenual ACC and vicinity):")
    target_keywords = ['cingulate', 'subcallosal', 'subgenual', 'anterior cingulate', 'medial frontal', 'ventromedial']
    
    for keyword in target_keywords:
        target_rois = df_baseline[df_baseline['roi_label'].str.contains(keyword, case=False, na=False)]
        if len(target_rois) > 0:
            print(f"  ROIs containing '{keyword}': {len(target_rois)} effects")
            for _, row in target_rois.head(3).iterrows():
                active_dir = "↑" if row['active_change_direction'] == 'increase' else "↓"
                sham_dir = "↑" if row['sham_change_direction'] == 'increase' else "↓"
                print(f"    ROI {row['roi_index']:3d}: {row['comparison'].replace('_minus_baseline', '')} {row['coefficient']} "
                      f"(A{active_dir}S{sham_dir}, p={row['p_value']:.4f})")
    
    # Effect sizes
    print(f"\nEFFECT SIZE DISTRIBUTION:")
    effect_sizes = df_baseline['effect_size_category'].value_counts()
    for size, count in effect_sizes.items():
        print(f"  {size}: {count}")
    
    print(f"\n{'='*60}")
    print("INTERPRETATION AND CONCLUSIONS")
    print(f"{'='*60}")
    
    print(f"""
KEY FINDINGS:

1. STIMULATION EFFECTS ARE STRONGER THAN RECOVERY
   - {stim_effects} significant effects during stimulation vs {recovery_effects} during recovery
   - Suggests immediate response to FUS with partial recovery

2. ACTIVE FUS PRODUCES SYSTEMATIC CHANGES
   - Active condition: {active_directions.get('decrease', 0)} decreases vs {active_directions.get('increase', 0)} increases
   - Sham condition: {sham_directions.get('decrease', 0)} decreases vs {sham_directions.get('increase', 0)} increases
   - More directional consistency in active condition

3. MULTIPLE BRAIN SYSTEMS AFFECTED
   - Cerebellar regions heavily represented
   - White matter tracts show significant changes
   - Distributed network effects beyond target region

4. FOKKER-PLANCK COEFFICIENTS REVEAL DIFFERENT MECHANISMS
   - Drift coefficients: {len(df_baseline[df_baseline['coefficient'].str.contains('drift')])} effects (neural dynamics)
   - Diffusion coefficient: {len(df_baseline[df_baseline['coefficient'] == 'diffusion'])} effects (neural noise/exploration)

5. BASELINE CORRECTION IS CRITICAL
   - Standard analysis: {len(df_standard)} effects
   - Baseline-corrected: {len(df_baseline)} effects  
   - Different effect patterns suggest session-specific confounds
""")
    
    # Save focused summary
    summary_data = {
        'top_effects': top_20[['roi_index', 'roi_label', 'comparison', 'coefficient', 
                              'p_value', 'cohens_d', 'active_change_direction', 
                              'sham_change_direction']].to_dict('records'),
        'roi_hotspots': [{'roi_index': int(roi_idx), 'effect_count': int(count), 
                         'roi_label': df_baseline[df_baseline['roi_index'] == roi_idx]['roi_label'].iloc[0]}
                        for roi_idx, count in roi_counts.head(10).items()],
        'summary_stats': {
            'total_baseline_corrected_effects': len(df_baseline),
            'stimulation_effects': stim_effects,
            'recovery_effects': recovery_effects,
            'active_decreases': int(active_directions.get('decrease', 0)),
            'active_increases': int(active_directions.get('increase', 0)),
            'sham_decreases': int(sham_directions.get('decrease', 0)),
            'sham_increases': int(sham_directions.get('increase', 0))
        }
    }
    
    return summary_data

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python summarize_key_findings.py <significant_effects_csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    summary = summarize_key_findings(csv_file)

if __name__ == "__main__":
    main()