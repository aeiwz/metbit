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
    affiliation: "1, 2"
affiliations:
  - name: Medical Biochemistry and Molecular Biology Graduate Study Program, Faculty of Medicine, Khon Kaen University, Khon Kaen, Thailand
    index: 1
  - name: kawa-technology, Independent Research & Development, Thailand
    index: 2
date: 28 May 2026
bibliography: paper.bib
---

# Summary

`metbit` is an open-source Python package for end-to-end nuclear magnetic resonance (NMR)-based metabolomics data analysis. It consolidates raw Bruker data processing, spectral preprocessing, normalization, peak alignment, multivariate statistical modeling, and interactive visualization into a single reproducible workflow. The package is designed for researchers who need to move from raw one-dimensional <sup>1</sup>H NMR spectra to interpretable statistical outputs without combining multiple incompatible commercial and open-source tools.

The current release of `metbit` provides automated digital-filter removal, Fourier transformation, phase correction, baseline correction, spectral normalization, interval-correlation-optimized shifting, Principal Component Analysis (PCA), Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA), permutation testing, Variable Importance in Projection (VIP) scoring, Statistical Total Correlation Spectroscopy (STOCSY), and browser-based graphical applications for peak picking and STOCSY exploration. Results are rendered as interactive `Plotly` figures, making routine inspection, reporting, and notebook-based analysis easier to reproduce and share [@Plotly2015].

# Statement of need

<sup>1</sup>H NMR spectroscopy is a cornerstone of untargeted metabolomics because it offers quantitative, non-destructive, and highly reproducible measurement of complex biological mixtures [@Emwas2019]. In practice, however, the route from raw free-induction decay files to a biological interpretation remains computationally fragmented. A typical study may require vendor software for acquisition export, separate scripts for phase and baseline correction, additional tools for normalization and alignment, and a different statistical environment for chemometric modeling and visualization.

This fragmentation creates several barriers. First, it makes routine analysis difficult to reproduce because important preprocessing decisions are often distributed across graphical software, spreadsheets, and ad hoc scripts. Second, it increases the training burden for laboratories that need to combine NMR spectroscopy with modern machine-learning workflows. Third, it limits interoperability with Python-based scientific computing tools that are now widely used for data science, statistics, and reproducible research [@Harris2020; @Pedregosa2011].

`metbit` addresses these barriers by providing a scriptable Python interface for the major steps in an NMR metabolomics workflow. The package is intended for metabolomics researchers, analytical chemists, bioinformaticians, and data scientists who need a coherent pipeline for preprocessing, modeling, and visualization. By exposing the workflow through Python classes and functions, `metbit` allows analyses to be version controlled, executed in notebooks or scripts, and integrated with established libraries such as `NumPy`, `SciPy`, `pandas`, and `scikit-learn` [@Virtanen2020; @pandas2020; @Pedregosa2011].

# State of the field

Several mature tools support parts of the metabolomics and NMR analysis workflow. `nmrglue` provides low-level Python support for reading and manipulating NMR data formats [@Helmus2013]. `pybaselines` offers a broad collection of baseline correction algorithms [@Erb2022]. Web platforms such as MetaboAnalyst provide accessible statistical analysis for metabolomics datasets [@Pang2022]. These tools are valuable, but they typically focus on either data access, an isolated preprocessing task, or downstream statistical analysis.

`metbit` differs by emphasizing workflow integration. It connects NMR-specific preprocessing steps with chemometric normalization, alignment, multivariate modeling, validation, and interactive visualization. This integrated design reduces format conversion overhead and helps users keep preprocessing choices, model settings, figures, and derived outputs in one reproducible Python environment.

# Functionality

`metbit` is organized around the natural sequence of NMR metabolomics analysis (Figure \ref{fig:workflow}). Each processing stage is intended to produce outputs that can be consumed directly by later stages, supporting both exploratory analysis and scripted production workflows.

![Detailed data-analysis workflow of the `metbit` package. The pipeline covers input, preprocessing, normalization, peak alignment, statistical modeling, and visualization. \label{fig:workflow}](figures/workflow_diagram.png)

The main package capabilities include:

- **NMR preprocessing:** conversion of raw Bruker FID directories into frequency-domain spectra, digital-filter removal, zero filling, Fourier transformation, automated phase correction, baseline correction, and chemical-shift calibration [@Chen2002; @Baek2015; @Zhang2010].
- **Spectral normalization and alignment:** Probabilistic Quotient Normalization (PQN), Multiplicative Scatter Correction (MSC), and interval-correlation-optimized shifting for managing concentration effects and peak displacement across samples [@Dieterle2006; @Martens1991; @Savorani2010].
- **Multivariate modeling:** PCA and OPLS-DA implementations for exploratory analysis and supervised classification, including cross-validation, permutation testing, and VIP scoring [@Trygg2002; @Wold1993].
- **Statistical spectroscopy:** STOCSY analysis for identifying co-varying resonances and supporting metabolite interpretation [@Nicholson2005; @Wishart2022].
- **Interactive visualization:** high-level plotting functions built on `Plotly`, with local `Dash` applications for STOCSY navigation and peak picking [@Plotly2015; @Dash2017].

# Implementation

`metbit` is implemented in Python and builds on the scientific Python ecosystem, including `NumPy`, `SciPy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `Plotly`, `Dash`, `nmrglue`, and `pybaselines` [@Harris2020; @Virtanen2020; @pandas2020; @Pedregosa2011; @Hunter2007; @Waskom2021; @Seabold2010; @Plotly2015; @Dash2017; @Helmus2013; @Erb2022].

The package uses tabular data structures for the main exchange format, allowing spectra and metadata to be joined, filtered, and passed into modeling functions. This design makes `metbit` compatible with common notebook workflows and enables interoperability with broader machine-learning pipelines. The OPLS-DA functionality includes model validation utilities and diagnostic visualizations, while plotting functions return interactive figures that can be inspected in notebooks or exported for reporting.

# Results and validation

In a typical analysis, a user can process raw Bruker data, normalize spectra, align peaks, fit PCA or OPLS-DA models, inspect VIP scores, and explore STOCSY correlations within a single Python session. This reduces the need to transfer intermediate files among vendor software, spreadsheets, and separate statistical environments.

The package has been exercised on datasets ranging from small pilot studies to medium-scale cohorts. In the manuscript benchmark, OPLS-DA fitting for a representative 500-sample by 50,000-variable dataset completed in under three seconds on a standard multi-core workstation. The project also includes automated tests and documentation to support ongoing maintenance and reproducible use.

# Availability

`metbit` is released under the MIT license. Source code is available at <https://github.com/aeiwz/metbit>, documentation is available at <https://metbit-docs.vercel.app>, and the package can be installed from PyPI with:

```bash
pip install metbit
```

# Author contributions

Theerayut Bubpamala: conceptualization, software design, implementation, testing, formal analysis, visualization, project administration, original manuscript drafting, and manuscript review and editing.

# AI usage disclosure

Generative AI tools were used for auxiliary manuscript drafting, manuscript review, and code review. The author reviewed and validated the generated material, accepts full responsibility for the final manuscript, and does not list these systems as authors because they cannot accept accountability for the work.

# Acknowledgements

The author thanks the open-source scientific Python community for the libraries on which `metbit` depends.

# Funding and conflicts of interest

`metbit` is independently developed and maintained by kawa-technology. This work received no external funding. The author declares no conflicts of interest.

# References
