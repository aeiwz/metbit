---
title: 'metbit: An Integrated Python Package for End-to-End NMR-Based Metabolomics Data Analysis'
tags:
  - Python
  - NMR
  - metabolomics
  - chemometrics
  - OPLS-DA
  - PCA
authors:
  - name: Theerayut Bubpamala
    orcid: 0000-0001-5176-5853
    corresponding: true
    affiliation: 1
affiliations:
  - name: kawa-technology, Independent Research & Development, Thailand
    index: 1
date: 23 April 2026
bibliography: paper.bib
---

# Summary

`metbit` is an open-source Python package designed to consolidate the fragmented NMR-based metabolomics workflow into a single, reproducible pipeline. It provides a complete end-to-end ecosystem, starting from raw vendor data (Bruker FID files) and proceeding through digital-filter removal, Fourier transformation, automated phase and baseline correction, spectral normalization, and peak alignment. For downstream analysis, `metbit` implements production-quality multivariate statistical models, including Principal Component Analysis (PCA) and Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA), with interactive visualizations and bundled web applications for data exploration.

# Statement of need

Proton nuclear magnetic resonance ($^1$H NMR) spectroscopy is an essential tool for systems biology, offering quantitative and non-destructive profiling of metabolites [@Emwas2019]. However, the journey from raw spectral acquisitions to interpretable biological models encompasses dozens of sequential computational steps. This complexity has historically restricted high-quality analysis to laboratories with expensive commercial licenses (e.g., Bruker TopSpin, Mnova) or researchers proficient in MATLAB and R [@Nicholson1999].

`metbit` fills a significant gap in the scientific Python ecosystem. While libraries like `nmrglue` [@Helmus2013] provide low-level bindings for vendor formats, they lack the integrated multivariate modeling and high-level visualization required for end-to-end metabolomics studies. By providing a coherent, class-based API built on foundational libraries like `NumPy` [@Harris2020] and `scikit-learn` [@Pedregosa2011], `metbit` empowers researchers to conduct reproducible, scriptable analyses that are natively interoperable with modern machine-learning and deep-learning frameworks.

# State of the field

Existing open-source tools typically address isolated segments of the metabolomics pipeline. The R package `nmr` and the web-based `MetaboAnalyst` [@Pang2022] are widely used but suffer from either steep learning curves or limited scripting extensibility. In the Python domain, `nmrglue` remains the standard for file reading, while `pybaselines` [@Erb2022] offers specialized baseline correction. `metbit` differentiates itself by unifying these capabilities into a linear workflow. It uses `pandas` DataFrames [@pandas2020] as its canonical data interchange format, ensuring that spectral data can be easily linked with clinical metadata or passed into scikit-learn Pipelines.

# Implementation and Features

`metbit` is organized into functionally cohesive modules reflecting the natural sequence of NMR processing (see Figure 1).

![Detailed data-analysis workflow of the metbit package. The pipeline covers six processing stages color-coded by category: navy (input), blue (preprocessing), dark teal (normalization), teal (peak alignment), purple (statistical modeling), and amber (visualization). \label{fig:workflow}](figures/workflow_diagram.png)

### Key Capabilities:
- **Signal Preprocessing**: Integrated pipeline for Fourier transformation, automated phase correction [@Chen2002], and multiple baseline estimation algorithms (AsLS, AirPLS, ArPLS).
- **Chemometric Normalization**: Robust implementations of Probabilistic Quotient Normalization (PQN) [@Dieterle2006] and Multiplicative Scatter Correction (MSC) [@Martens1991].
- **Advanced Modeling**: Validated PCA and OPLS-DA [@Trygg2002] implementations with cross-validation and sign-encoded Variable Importance in Projection (VIP) scoring.
- **Statistical Spectroscopy**: Interactive Statistical Total Correlation Spectroscopy (STOCSY) [@Nicholson2005] for metabolic pathway connectivity.
- **Interactivity**: All plots are rendered using `Plotly` [@Plotly2015], providing native hover, zoom, and pan capabilities. Two `Dash` [@Dash2017] applications are included for graphical peak picking and STOCSY navigation.

# AI usage disclosure

Generative AI tools (Claude 4.6 and GPT-5.2) were used for auxiliary support in drafting and reviewing the manuscript and code review. The corresponding author (T.B.) accepts full responsibility for the final content and has reviewed and validated all outputs.

# Acknowledgements

The author thanks the open-source scientific Python community for the libraries on which `metbit` depends.

# References
