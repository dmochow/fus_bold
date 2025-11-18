# Fokker-Planck Analysis of FUS-BOLD Data

This directory contains code for analyzing focused ultrasound (FUS) effects on brain activity using stochastic thermodynamic methods, specifically fitting the 1-D Fokker-Planck equation to BOLD time series data.

## Overview

The analysis examines how transcranial focused ultrasound stimulation affects the brain's free energy landscape by fitting Fokker-Planck equations to BOLD time series data from 16 subjects across three experimental periods:
- **Baseline** (TRs 0-299): Pre-stimulation period
- **Stimulation** (TRs 300-599): Active FUS or sham stimulation
- **Recovery** (TRs 600-899): Post-stimulation period

## Files

### Core Analysis Modules
- `fokker_planck.py` - Main Fokker-Planck fitting class and functions
- `batch_fokker_planck.py` - Batch processing for all subjects/ROIs
- `simple_analysis.py` - Statistical analysis of fitted coefficients

### Testing and Validation
- `quick_test.py` - Quick validation of fitting on sample data
- `test_fokker_planck.py` - Comprehensive testing suite
- `analyze_coefficients.py` - Analysis of fitted coefficients

### Visualization and Reporting
- `visualize_fp_results.py` - Create visualizations of results
- `comprehensive_analysis.py` - Full analysis with plots (resource intensive)

## Usage

### 1. Quick Test
```bash
python quick_test.py
```

### 2. Batch Processing (Small Test)
```bash
python batch_fokker_planck.py --subjects 2 --rois 10 --processes 2
```

### 3. Full Dataset Processing
```bash
python batch_fokker_planck.py --subjects 16 --rois 1024 --processes 8
```

### 4. Statistical Analysis
```bash
python simple_analysis.py fokker_planck_results_full.csv
```

## Key Results from Test Data

From analysis of 2 subjects × 10 ROIs:

### Drift Coefficients
- **drift_0** (constant term): Shows small differences between active and sham
- **drift_1** (linear term): Medium effect size during baseline (d=0.636, p=0.051)

### Diffusion Coefficients  
- **diffusion**: Consistently lower in active vs sham condition
- Medium effect size during stimulation (d=-0.552, p=0.089)

### Within-Condition Effects
- **Sham condition** shows significant changes from baseline to stimulation
- **Active condition** shows more stable coefficients across periods

## Interpretation

The Fokker-Planck analysis reveals:

1. **Energy Injection Hypothesis**: Active FUS appears to reduce diffusion coefficients, suggesting more constrained dynamics
2. **Sham Effects**: Sham stimulation shows significant changes, indicating procedural effects
3. **Recovery Patterns**: Both conditions show partial recovery toward baseline values

## Technical Details

### Fokker-Planck Equation
The fitted equation is: `∂P/∂t = -∂/∂x[D1(x)P] + (1/2)∂²/∂x²[D2(x)P]`

Where:
- `D1(x) = drift_0 + drift_1*x` (linear drift)
- `D2(x) = diffusion` (constant diffusion)

### Coefficient Interpretation
- **drift_0**: Baseline tendency toward equilibrium
- **drift_1**: State-dependent drift (negative = restoring force)
- **diffusion**: Noise strength/exploration rate

## Output Files

- `fokker_planck_results_full.csv` - All fitted coefficients
- `fokker_planck_statistics.csv` - Statistical comparisons
- Various visualization files (.png)

## Next Steps

1. **Scale to Full Dataset**: Process all 16 subjects × 1024 ROIs
2. **ROI-Specific Analysis**: Examine effects in target regions (subgenual ACC)
3. **Temporal Dynamics**: Analyze coefficient evolution during stimulation
4. **Network Analysis**: Examine inter-ROI coupling changes