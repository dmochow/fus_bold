# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neuroimaging research project examining the brain's response to transcranial focused ultrasound stimulation using fMRI BOLD imaging. The study investigates how 500 kHz ultrasound applied to the subgenual anterior cingulate cortex affects brain dynamics from a stochastic thermodynamic perspective.

## Experimental Design

- **Subjects**: 16 healthy participants
- **Sessions**: 2 per subject (active FUS vs sham control)
- **Duration**: 15 minutes per session (900 TRs at TR=1s)
- **Timeline**: 
  - TRs 0-299: Baseline period
  - TRs 300-599: Stimulation period (20 TRs on, 40 TRs off pattern)
  - TRs 600-899: Recovery period
- **Target**: Subgenual anterior cingulate cortex

## Data Structure

### Main Dataset
- **Location**: `data/precomputed/difumo_time_series.pkl`
- **Format**: Pickled dictionary with keys `'active'` and `'sham'`
- **Content**: Each key contains a list of 16 elements (one per subject)
- **Shape**: Each element is a 900×1024 numpy array (time × DiFuMo ROIs)
- **Preprocessing**: fMRIPrep with default options

### Raw Data
- **Location**: `data/resampled_bold_flywheel/`
- **Format**: BIDS-formatted fMRIPrep outputs
- **Content**: Preprocessed BOLD data, anatomical data, FreeSurfer outputs

## Analysis Goals

The primary analysis focuses on fitting the 1-D Fokker-Planck equation to BOLD time series data to:
1. Extract drift and diffusion coefficients for each subject/ROI/condition
2. Compare coefficients across baseline, stimulation, and recovery periods
3. Test hypothesis that ultrasound injection alters the brain's free energy landscape

## Development Approach

- Start with prototype code for Fokker-Planck fitting on small data samples
- Test and verify correctness before scaling to full dataset
- Structure code to handle all subjects/ROIs/conditions systematically
- Use notebooks for exploratory analysis and prototyping

## Key Technical Considerations

- Time series are 900 timepoints long at 1s TR
- DiFuMo atlas provides 1024 brain regions of interest
- Analysis should be performed separately for each experimental period
- Computational efficiency important given large dataset size (16 subjects × 1024 ROIs × 2 conditions × 3 periods)