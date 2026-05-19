# Exact DMD and Time-delay DMD for Dynamical Systems

This project implements Exact Dynamic Mode Decomposition (DMD) and Time-delay DMD in Python, and compares their reconstruction and prediction performance on synthetic dynamical systems.

Project Overview

Dynamic Mode Decomposition (DMD) is a data-driven method for analyzing high-dimensional time-dependent data. It approximates the evolution between consecutive snapshots using a linear operator and extracts dynamic modes and eigenvalues.

In this project, I implemented:

Exact DMD on a linear oscillator dataset Exact DMD on a synthetic spatio-temporal dataset
Time-delay DMD as an extension method
Reconstruction and short-term prediction experiments
Rank truncation analysis
 Relative Frobenius norm error evaluation

Motivation

The goal of this project is to understand how DMD can extract dominant dynamic structures from time-series data, and how time-delay embedding can improve reconstruction performance for more complex spatio-temporal dynamics.

Methods

The project includes the following steps:

1. Generate synthetic dynamical datasets
2. Construct snapshot matrices
3. Apply truncated SVD
4. Compute the reduced DMD operator
5. Extract DMD eigenvalues and modes
6. Reconstruct observed data
7. Perform short-term prediction
8. Compare Exact DMD and Time-delay DMD using relative reconstruction and prediction errors

Key Results

- Exact DMD works well on the simple linear oscillator dataset.
- On the spatio-temporal dataset, Exact DMD gives high reconstruction error.
- Time-delay DMD significantly improves reconstruction performance.
- Rank truncation affects reconstruction quality, especially for Time-delay DMD.
- Prediction remains more challenging than reconstruction on the spatio-temporal dataset.

Example Results

| Method | Reconstruction Error | Prediction Error |
| Exact DMD on spatio-temporal data | 1.0465 | 1.0000 |
| Time-delay DMD on spatio-temporal data | 0.0267 | 0.5855 |
 Repository Structure

text
dmd-time-delay-dmd/
├── README.md
├── src/
│   └── dmd_demo.py
├── figures/
├── reports/
│   └── dmd_project_report_redacted.pdf
├── requirements.txt
└── .gitignore
