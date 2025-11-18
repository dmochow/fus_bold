import numpy as np
from nilearn import datasets
import re

def find_subgenual_rois():
    """Find DiFuMo ROIs closest to subgenual anterior cingulate cortex."""
    
    print("Loading DiFuMo atlas...")
    difumo = datasets.fetch_atlas_difumo(dimension=1024)
    labels = difumo.labels
    
    print(f"Searching through {len(labels)} DiFuMo ROIs...")
    
    # Keywords related to subgenual anterior cingulate
    subgenual_keywords = [
        'subgenual', 'subcallosal', 'ventral anterior cingulate', 
        'rostral anterior cingulate', 'perigenual'
    ]
    
    # General anterior cingulate keywords
    anterior_cingulate_keywords = [
        'anterior cingulate', 'cingulate anterior', 'paracingulate anterior',
        'cingulate gyrus anterior', 'ACC'
    ]
    
    # Broader cingulate keywords
    cingulate_keywords = [
        'cingulate', 'paracingulate', 'cingulum'
    ]
    
    # Medial prefrontal keywords (anatomically close)
    medial_pfc_keywords = [
        'medial prefrontal', 'ventromedial prefrontal', 'orbitofrontal medial',
        'frontal pole medial', 'medial orbitofrontal'
    ]
    
    print("\n=== EXACT SUBGENUAL/SUBCALLOSAL MATCHES ===")
    subgenual_matches = []
    for i, label in enumerate(labels):
        label_str = str(label).lower()
        for keyword in subgenual_keywords:
            if keyword in label_str:
                subgenual_matches.append((i, str(label), keyword))
                print(f"ROI {i:3d}: {str(label)}")
                break
    
    print(f"\nFound {len(subgenual_matches)} exact subgenual/subcallosal matches")
    
    print("\n=== ANTERIOR CINGULATE MATCHES ===")
    acc_matches = []
    for i, label in enumerate(labels):
        label_str = str(label).lower()
        # Skip if already found in subgenual matches
        if any(i == match[0] for match in subgenual_matches):
            continue
            
        for keyword in anterior_cingulate_keywords:
            if keyword in label_str:
                acc_matches.append((i, str(label), keyword))
                print(f"ROI {i:3d}: {str(label)}")
                break
    
    print(f"\nFound {len(acc_matches)} anterior cingulate matches")
    
    print("\n=== GENERAL CINGULATE MATCHES (first 20) ===")
    general_cingulate_matches = []
    for i, label in enumerate(labels):
        label_str = str(label).lower()
        # Skip if already found
        if any(i == match[0] for match in subgenual_matches + acc_matches):
            continue
            
        for keyword in cingulate_keywords:
            if keyword in label_str:
                general_cingulate_matches.append((i, str(label), keyword))
                break
    
    # Show first 20 general cingulate matches
    for match in general_cingulate_matches[:20]:
        print(f"ROI {match[0]:3d}: {match[1]}")
    
    if len(general_cingulate_matches) > 20:
        print(f"... and {len(general_cingulate_matches) - 20} more cingulate ROIs")
    
    print(f"\nFound {len(general_cingulate_matches)} total cingulate matches")
    
    print("\n=== MEDIAL PREFRONTAL CORTEX MATCHES (anatomically adjacent) ===")
    mpfc_matches = []
    for i, label in enumerate(labels):
        label_str = str(label).lower()
        for keyword in medial_pfc_keywords:
            if keyword in label_str:
                mpfc_matches.append((i, str(label), keyword))
                print(f"ROI {i:3d}: {str(label)}")
                break
    
    print(f"\nFound {len(mpfc_matches)} medial PFC matches")
    
    # Search for specific anatomical terms that might be relevant
    print("\n=== OTHER POTENTIALLY RELEVANT REGIONS ===")
    other_keywords = [
        'ventral', 'rostral', 'medial frontal', 'frontal medial',
        'orbitofrontal', 'frontopolar'
    ]
    
    other_matches = []
    for i, label in enumerate(labels):
        label_str = str(label).lower()
        # Skip if already found
        already_found = any(i == match[0] for match in 
                          subgenual_matches + acc_matches + general_cingulate_matches + mpfc_matches)
        if already_found:
            continue
            
        # Look for ventral/rostral regions in frontal areas
        if ('ventral' in label_str or 'rostral' in label_str) and 'frontal' in label_str:
            other_matches.append((i, str(label), 'ventral/rostral frontal'))
            print(f"ROI {i:3d}: {str(label)}")
        elif 'orbitofrontal' in label_str and 'medial' in label_str:
            other_matches.append((i, str(label), 'medial orbitofrontal'))
            print(f"ROI {i:3d}: {str(label)}")
    
    print(f"\nFound {len(other_matches)} other potentially relevant regions")
    
    # Create summary
    print("\n" + "="*80)
    print("SUMMARY OF SUBGENUAL ACC CANDIDATE ROIs")
    print("="*80)
    
    print(f"\n1. HIGHEST PRIORITY (exact subgenual/subcallosal): {len(subgenual_matches)} ROIs")
    for match in subgenual_matches:
        print(f"   ROI {match[0]:3d}: {match[1]}")
    
    print(f"\n2. HIGH PRIORITY (anterior cingulate): {len(acc_matches)} ROIs")
    for match in acc_matches[:10]:  # Show first 10
        print(f"   ROI {match[0]:3d}: {match[1]}")
    if len(acc_matches) > 10:
        print(f"   ... and {len(acc_matches) - 10} more")
    
    print(f"\n3. MEDIUM PRIORITY (general cingulate): {len(general_cingulate_matches)} ROIs")
    print("   (Too many to list - see detailed output above)")
    
    print(f"\n4. ANATOMICALLY ADJACENT (medial PFC): {len(mpfc_matches)} ROIs")
    for match in mpfc_matches:
        print(f"   ROI {match[0]:3d}: {match[1]}")
    
    # Return the most relevant ROI indices
    priority_rois = {
        'subgenual_exact': [match[0] for match in subgenual_matches],
        'anterior_cingulate': [match[0] for match in acc_matches],
        'general_cingulate': [match[0] for match in general_cingulate_matches],
        'medial_pfc': [match[0] for match in mpfc_matches]
    }
    
    return priority_rois, labels

def analyze_subgenual_results(json_file, priority_rois):
    """Analyze results specifically for subgenual-related ROIs."""
    
    import json
    import pandas as pd
    
    print(f"\nAnalyzing Fokker-Planck results for subgenual-related ROIs...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    roi_results = data['roi_results']
    
    # Collect results for priority ROIs
    subgenual_results = []
    
    for category, roi_indices in priority_rois.items():
        print(f"\n=== {category.upper().replace('_', ' ')} RESULTS ===")
        
        category_results = []
        for roi_idx in roi_indices:
            roi_key = f"roi_{roi_idx:04d}"
            
            if roi_key in roi_results:
                roi_data = roi_results[roi_key]
                roi_label = roi_data['roi_label']
                
                for period in ['baseline', 'stimulation', 'recovery']:
                    for coeff in ['drift_0', 'drift_1', 'diffusion']:
                        coeff_data = roi_data['periods'][period]['coefficients'][coeff]
                        
                        if 'p_value' in coeff_data:
                            result = {
                                'category': category,
                                'roi_index': roi_idx,
                                'roi_label': roi_label,
                                'period': period,
                                'coefficient': coeff,
                                'p_value': coeff_data['p_value'],
                                'cohens_d': coeff_data.get('cohens_d', np.nan),
                                'mean_difference': coeff_data.get('mean_difference', np.nan)
                            }
                            category_results.append(result)
                            subgenual_results.append(result)
        
        # Show significant results for this category
        category_df = pd.DataFrame(category_results)
        if len(category_df) > 0:
            significant = category_df[category_df['p_value'] < 0.05].sort_values('p_value')
            
            if len(significant) > 0:
                print(f"Significant results (p < 0.05): {len(significant)}")
                for _, row in significant.head(5).iterrows():
                    print(f"  ROI {row['roi_index']:3d} ({row['period']}-{row['coefficient']}): "
                          f"p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
                    print(f"      {row['roi_label'][:60]}...")
            else:
                print("No significant results found")
        else:
            print("No results found for this category")
    
    # Overall summary
    df_subgenual = pd.DataFrame(subgenual_results)
    if len(df_subgenual) > 0:
        total_significant = len(df_subgenual[df_subgenual['p_value'] < 0.05])
        print(f"\n=== OVERALL SUBGENUAL REGION SUMMARY ===")
        print(f"Total tests: {len(df_subgenual)}")
        print(f"Significant results (p < 0.05): {total_significant} ({total_significant/len(df_subgenual)*100:.1f}%)")
        
        if total_significant > 0:
            print(f"\nTop 10 most significant subgenual-related results:")
            top_significant = df_subgenual[df_subgenual['p_value'] < 0.05].nsmallest(10, 'p_value')
            for i, (_, row) in enumerate(top_significant.iterrows(), 1):
                print(f"{i:2d}. ROI {row['roi_index']:3d} ({row['category']}) - "
                      f"{row['period']}-{row['coefficient']}: p={row['p_value']:.4f}, d={row['cohens_d']:.3f}")
        
        # Save subgenual-specific results
        output_path = "/Users/jacekdmochowski/PROJECTS/fus_bold/code/subgenual_roi_results.csv"
        df_subgenual.to_csv(output_path, index=False)
        print(f"\nSubgenual ROI results saved to: subgenual_roi_results.csv")
    
    return df_subgenual

def main():
    """Main function."""
    import sys
    
    # Find subgenual ROIs
    priority_rois, labels = find_subgenual_rois()
    
    # If JSON file provided, analyze results
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        df_results = analyze_subgenual_results(json_file, priority_rois)

if __name__ == "__main__":
    main()