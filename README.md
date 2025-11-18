# Dmochowski Lab: tFUS-fMRI Project

## Objective
- Identify the effects of low intensity transcranial focused ultrasound (tFUS) stimulation on human brain activity measured with functional Magnetic Resonance Imaging (fMRI) Blood-Oxygen Level Dependent (BOLD).
- Cast tFUS in the framework of stochastic thermodynamics.

## Data
- Too big for GitHub, key data structures are on [Dropbox](https://www.d ropbox.com/scl/fo/p8xonn1wj6u3zmaomz2s6/AILrK4NfFAFUl1OhFIjGl50?rlkey=f4v4twth7uolnzitnte7xc3nt&dl=0)
- Source data is on the cloud via the Flywheel platform.

## Powerpoint Slide Deck
- [NANS 2025](./ppt/NANS25SpeakerTemplate730_.pptx)

## Notebooks
- [Basic data exploration](./code/notebooks/explore_difumo_ts.ipynb)

## Papers
- [Deco et al. 2025 Cost of Cognition](./papers/deco_2025.06.18.660368v2.full.pdf)


## source path:
example 1: 
/Volumes/Backup Plus/2023 MacBook Pro 091525/jacekdmochowski/PROJECTS/fus/data/resampled_bold_flywheel/resampled_bold_sub-BOYER_ses-ACTIVE--2024-09-27/input/66ee0c9c701c35f5e44e91b5/sub-BOYER/ses-ACTIVE/anat/sub-BOYER_ses-ACTIVE_desc-preproc_T1w.nii.gz

example 2: 
/Volumes/Backup Plus/2023 MacBook Pro 091525/jacekdmochowski/PROJECTS/fus/data/resampled_bold_flywheel/resampled_bold_sub-ADLER_ses-SHAM--2024-09-13/input/66cc6be7788844bef1926b31/sub-ADLER/ses-SHAM/anat/sub-ADLER_ses-SHAM_run-01_desc-preproc_T1w.nii.gz

## destination path:
/Volumes/jacekdmochowski/PROJECTS/fus/data/all_anatomicals

## technical details
- embs data was analyzed with load_and_prepare_data() as in this notebook https://github.com/dmochow/fus/blob/main/code/old/old_notebooks/analyze_resampled_preprocessed_bold.ipynb
- we also generate "minimal" and "conservative" preprocessed datasets (again taking the fmriprep as starting point) here https://github.com/dmochow/fus/blob/main/code/minimal_preprocessing_pipeline.py