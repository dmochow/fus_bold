
# Independent Study Syllabus (Fall 2025)

**Title:** *Thermodynamic Modeling of BOLD fMRI During Focused Ultrasound Neuromodulation*  
**Mentor:** Prof. Dmochowski
**Student:** Andrew Birnbaum 
**Credits:** 3

---

## 🎯 Goal

Develop and validate methods for estimating free energy, entropy production, and entropy flow from BOLD fMRI time series collected during tFUS in humans.

---

## 🧭 Learning Objectives

By the end of the semester, the student will be able to:

- Understand and implement stochastic modeling of fMRI time series (e.g., drift/diffusion, Fokker–Planck, Langevin models).
- Estimate and visualize thermodynamic quantities such as entropy production and free energy from BOLD signals.
- Apply statistical comparisons across experimental conditions (pre-, during-, post-tFUS).
- Deliver research-grade code and documentation for reuse in grant proposals and manuscripts.

---

## 📅 Weekly Breakdown

### Weeks 1–2: Orientation + Literature + Setup
- Read key papers (Deco, Sengupta, Wolpert et al.)
- Set up Python environment (Nilearn, Nibabel, NumPy, etc.)
- Explore sample BOLD fMRI data  
**Deliverable:** Background summary (1–2 pages)

---

### Weeks 3–5: Data Access + Signal Processing
- Preprocess and segment time series by condition
- Compute basic signal statistics
- Begin estimating drift and diffusion coefficients  
**Deliverable:** Preprocessed time series + exploratory plots

---

### Weeks 6–8: Drift/Diffusion Modeling
- Implement drift/diffusion estimation
- Validate using synthetic time series
- Explore voxel-wise vs ROI-wise methods  
**Deliverable:** Estimator notebook

---

### Weeks 9–10: Entropy Production Estimation
- Estimate entropy production with Fokker–Planck equation
- Apply across fMRI conditions  
**Deliverable:** Code for thermodynamic metrics

---

### Weeks 11–12: Visualization + Statistical Comparison
- Compare conditions with statistical tests
- Visualize entropy production  metrics  
**Deliverable:** Slide deck or figures

---

### Weeks 13–14: Integration and Documentation
- Clean up and modularize code
- Prepare usage documentation and examples  
**Deliverable:** Pipeline + documentation

---

### Week 15: Final Report + Presentation
- Present project findings
- Submit 6–8 page final report or Jupyter-style write-up

---

## 📦 Deliverables Summary

| Week | Deliverable |
|------|-------------|
| 2    | Background summary |
| 5    | Preprocessed time series |
| 8    | Drift/diffusion estimator |
| 10   | Thermodynamic metrics code |
| 12   | Visualizations and plots |
| 14   | Documented codebase |
| 15   | Final report + presentation |

---

## 🧰 Tools & Libraries

- **Python**: NumPy, SciPy, Matplotlib, Pandas, Scikit-learn, Nibabel, Nilearn
- **fMRI Preprocessing**: Assumes data already preprocessed
- **Version control**: GitHub repo for reproducibility and collaboration
