# metbit: An Integrated Python Package for End-to-End NMR-Based Metabolomics Data Analysis

---

**Authors**

Theerayut Bubpamala¹\*

¹ kawa-technology, Independent Research & Development

\* Corresponding author: theerayut_aeiw_123@hotmail.com  
  GitHub: https://github.com/aeiwz/metbit  
  PyPI: https://pypi.org/project/metbit/

---

![Graphical Abstract](figures/graphical_abstract.png)

**Figure GA.** Graphical abstract summarizing the five-stage metbit pipeline: raw Bruker NMR data input, signal preprocessing, spectral normalization and peak alignment, statistical modeling (PCA, OPLS-DA, STOCSY), and interactive output generation.

---

## Abstract

Nuclear magnetic resonance (NMR) spectroscopy remains one of the most information-rich platforms for untargeted metabolomics, yet the computational tools required to transform raw spectral acquisitions into interpretable biological conclusions are fragmented across incompatible software ecosystems. We present **metbit** (version 8.7.7), an open-source Python package that consolidates the complete NMR metabolomics analytical workflow into a single, pip-installable library. Starting from raw Bruker FID files, metbit delivers a sequentially integrated pipeline covering digital-filter removal, zero-filling, Fourier transformation, automated phase correction, baseline correction (including asymmetric least-squares, adaptive iteratively reweighted penalized least-squares, and rubberband methods), chemical-shift calibration, spectral normalization (probabilistic quotient normalization, standard normal variate, and multiplicative scatter correction), peak alignment via interval-correlation-optimized shifting, peak detection and multiplet classification, and statistical spectroscopy through Statistical Total Correlation Spectroscopy (STOCSY). Downstream of preprocessing, metbit provides production-quality implementations of Principal Component Analysis (PCA) and Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA), with cross-validation, permutation testing, Variable Importance in Projection (VIP) scoring, S-plot generation, and time-trajectory visualization. All outputs are rendered as interactive Plotly figures, and two Dash-based graphical applications are bundled for STOCSY exploration and interactive peak picking. metbit is released under the MIT license and is freely available at https://github.com/aeiwz/metbit.

**Keywords:** metabolomics; NMR spectroscopy; chemometrics; OPLS-DA; PCA; Python; open-source bioinformatics

---

## 1. Introduction

Metabolomics, the global profiling of small-molecule metabolites in biological samples, has emerged as an indispensable systems-biology tool for understanding disease mechanisms, drug effects, and environmental perturbations (Nicholson et al., 1999; Wishart et al., 2022). Among the analytical platforms available to metabolomics researchers, proton nuclear magnetic resonance (¹H NMR) spectroscopy holds a distinctive position by virtue of its quantitative accuracy, non-destructive sample handling, structural elucidation capacity, and broad compound coverage without prior chromatographic separation (Emwas et al., 2019). However, the journey from raw NMR free-induction decay (FID) files to biologically interpretable multivariate models encompasses more than a dozen sequential computational steps, each requiring careful parameter selection and quality assessment. This workflow complexity historically limited NMR metabolomics to laboratories with access to expensive commercial software such as Bruker TopSpin, Mnova (Mestrelab Research), or AMIX, or to end-users proficient in programming environments such as MATLAB and R.

The open-source landscape has responded with a range of tools that address individual stages of the NMR metabolomics pipeline. NMRglue provides low-level Python bindings for reading and processing vendor NMR formats (Helmus and Jaroniec, 2013). The R package nmr provides spectrum processing utilities within the Bioconductor framework, while MetaboAnalyst (Pang et al., 2022) offers a comprehensive web-based platform with a graphical interface but limited scripting extensibility. BATMAN (Bayesian automated metabolite analyzer), SRV (statistical recoupling of variables), and ASICS provide specialized functionality for targeted assignment and quantification. Despite these valuable contributions, no single freely available Python package currently integrates the full NMR metabolomics workflow (from raw file reading through spectral preprocessing, normalization, alignment, and multivariate statistical modeling) in a manner that is scriptable, reproducible, and natively interactive without requiring external graphical interfaces.

metbit was developed to fill this gap. The package provides a coherent, end-to-end Python application programming interface (API) for NMR-based metabolomics that reduces the number of software dependencies a researcher must manage, promotes reproducible scripted analysis, and delivers publication-quality interactive visualizations. In this manuscript, we describe the design principles, implementation, and functional capabilities of metbit, and illustrate how its integrated pipeline accelerates the standard NMR metabolomics workflow.

---

## 2. Implementation

### 2.1 Architecture and Design Principles

metbit is organized as a pure Python package composed of functionally cohesive modules (Figure 1). The top-level namespace, exposed through a single `import metbit` statement, re-exports all primary public classes and functions, eliminating the need for users to navigate the internal module hierarchy for routine analyses. The package requires Python ≥ 3.10 and depends exclusively on widely maintained scientific Python libraries: NumPy (Harris et al., 2020), pandas (McKinney, 2010; The pandas Development Team, 2020), SciPy (Virtanen et al., 2020), scikit-learn (Pedregosa et al., 2011), Plotly (Plotly Technologies, 2015), Dash (Plotly Technologies, 2017), matplotlib (Hunter, 2007), seaborn (Waskom, 2021), statsmodels (Seabold and Perktold, 2010), pingouin (Vallat, 2018), NMRglue (Helmus and Jaroniec, 2013), pybaselines (Erb, 2022), pyChemometrics, lingress, and tqdm (da Costa-Luis et al., 2023). All dependencies are automatically resolved upon installation via pip.

The design follows three guiding principles. First, **linear workflow coherence**: modules are ordered to reflect the natural sequence of NMR data processing, and outputs from each stage are designed as valid inputs to the next. Second, **stateless class-based interfaces**: analytical objects (e.g., `pca`, `opls_da`, `nmr_preprocessing`) encapsulate both the fitted model parameters and the visualization methods, so that a complete analysis (including all diagnostic plots) can be reproduced from a single persistent object. Third, **progressive disclosure**: sensible defaults make the package immediately useful to non-specialists, while every parameter is accessible for expert customization.

![Workflow Diagram](figures/workflow_diagram.png)

**Figure 1.** Detailed data-analysis workflow of the metbit package. Boxes represent the six processing stages colour-coded by functional category (navy: data input; blue: signal preprocessing; dark teal: normalization; teal: peak alignment; purple: statistical modeling; amber: visualization and output). Dependency badges in the upper-right corner of each box identify the primary Python library invoked at that stage. The workflow forks at Stage ⑤ to accommodate both STOCSY correlation spectroscopy (left branch; STOCSY.py) and multivariate modeling via PCA/OPLS-DA (right branch; metbit.py), with both branches converging at the interactive visualization and output layer. The diagram was generated programmatically using matplotlib (Hunter, 2007) and is fully reproducible from `manuscript/figures/generate_workflow.py`.

### 2.2 NMR Preprocessing Pipeline

The `nmr_preprocessing` class implements a complete single-file and batch NMR preprocessing pipeline integrated around NMRglue (Helmus and Jaroniec, 2013). Starting from raw Bruker FID directory trees, the preprocessing workflow proceeds through the following stages.

**Digital filter removal.** Modern Bruker spectrometers apply a digital filter during acquisition to suppress out-of-band noise. The initial processing step calls `ng.bruker.remove_digital_filter()` to remove the filter-induced phase roll before Fourier transformation, restoring the correct spectral baseline at the edges of the spectral width.

**Zero-filling and Fourier transformation.** Zero-filling extends the FID to a power-of-two length, improving the digital resolution of the resulting frequency-domain spectrum. The discrete Fourier transform is then applied to convert time-domain data to chemical-shift space, yielding the familiar NMR spectral profile.

**Automated phase correction.** Phase errors arise from timing imperfections and group delays and manifest as absorptive/dispersive distortions across the spectrum. metbit applies `ng.process.proc_autophase.autops()` with the peak-minima algorithm by default (Chen et al., 2002), with zero-order and first-order phase parameters adjustable by the user.

**Baseline correction.** The `baseline` module provides six baseline estimation algorithms accessed through a unified `baseline_correct()` function and a convenience `bline()` wrapper. Implemented methods include asymmetric least-squares smoothing (AsLS; Eilers and Boelens, 2005), adaptive iteratively reweighted penalized least-squares (AirPLS; Zhang et al., 2010), asymmetrically reweighted penalized least-squares (ArPLS; Baek et al., 2015), modified polynomial fitting (ModPoly), improved modified polynomial fitting (IModPoly), and a pure-Python rubberband baseline based on the lower convex hull. Import errors for optional pybaselines methods are handled gracefully with informative fallbacks, ensuring robustness across installation environments.

**Calibration.** The `calibrate` module aligns spectra to a user-specified internal reference compound (e.g., 3-(trimethylsilyl)propanoic-2,2,3,3-d4 acid sodium salt, TSP, δ = 0.00 ppm in aqueous samples; or 4,4-dimethyl-4-silapentane-1-sulfonic acid, DSS) by identifying the reference peak and applying a global chemical-shift offset.

**Spectral normalization.** Following baseline correction and calibration, systematic variation in sample concentration or amount must be corrected before multivariate analysis. The `Normalization` class in `spec_norm.py` implements three established normalization strategies. Probabilistic Quotient Normalization (PQN; Dieterle et al., 2006) divides each spectrum by the median of the fold-changes relative to a reference (the median spectrum), making it robust to large spectral differences. Standard Normal Variate (SNV) normalization applies column-wise mean centering and scaling, removing multiplicative baseline effects. Multiplicative Scatter Correction (MSC; Martens and Stark, 1991) corrects for differences in light scattering by regressing each spectrum against the mean spectrum. The `Normalise` class additionally exposes a higher-level wrapper with method dispatch via a string argument.

### 2.3 Peak Alignment

Chemical shift drift between samples, caused by pH variation, temperature fluctuations, or ionic strength differences, can severely degrade multivariate model quality. The `alignment` module addresses this with three components. The `detect_multiplets()` function identifies and classifies multiplet patterns (singlet, doublet, triplet, quartet, or multiplet) using peak-finding via `scipy.signal.find_peaks`, equal-spacing tests in Hz, and binomial coefficient height-ratio matching. The `icoshift_align()` function implements interval-correlation-optimized shifting (icoshift; Savorani et al., 2010), which partitions the spectrum into user-defined or automatically determined intervals and shifts each interval independently to maximize cross-correlation with a reference spectrum. The `PeakAligner` convenience class combines detection and alignment, exposing a scikit-learn-compatible `fit_transform()` interface.

### 2.4 STOCSY Analysis

Statistical Total Correlation Spectroscopy (STOCSY; Nicholson et al., 2005) identifies resonances that co-vary across a sample set, revealing metabolic pathway connectivity without prior metabolite identification. The `STOCSY()` function computes Pearson correlation coefficients and corresponding p-values between an anchor resonance (specified by its ppm value) and all other spectral points. Significance is assessed against a user-configurable p-value threshold (default *p* < 0.0001). Results are rendered as an interactive Plotly scatter plot in which point color encodes the correlation coefficient and opacity encodes significance. A companion Dash web application (`STOCSY_app`) provides a self-contained browser-based STOCSY explorer requiring no coding on the part of the end user.

### 2.5 Multivariate Statistical Modeling

#### 2.5.1 Principal Component Analysis

The `pca` class wraps scikit-learn's `PCA` decomposition within a metabolomics-oriented interface. The class applies a selectable scaling transformation before decomposition, with four options: unit-variance (autoscaling, power = 1), Pareto scaling (power = 0.5), mean centering (power = 0), and min-max scaling, implemented in the `Scaler` class. Pareto scaling is the default, as it reduces the influence of high-variance noise signals while preserving relevant metabolite variance (van den Berg et al., 2006). After fitting, the class computes explained-variance ratios for each principal component and a Q² reconstruction statistic estimated on a held-out test partition. The public API exposes seven visualization methods: `plot_observe_variance()` (bar chart of per-component R²X), `plot_cumulative_observed()` (combined bar and line chart of cumulative R²X), `plot_pca_scores()` (2D scores scatter with confidence ellipses), `plot_3d_pca()` (3D interactive scores plot), `plot_pca_trajectory()` (mean ± SEM time-series trajectory on PCA axes), and `plot_loading_()` (loadings scatter plotted against chemical shift). All scores plots support user-supplied color and symbol mappings for groups and time points, with automatic palette generation from the full Plotly qualitative color library when no mapping is provided. Hotelling's T² confidence ellipses at 95% confidence are rendered for each group or for the full dataset using a parametric ellipse computed from the eigendecomposition of the sample covariance matrix.

#### 2.5.2 OPLS-DA

Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA; Trygg and Wold, 2002) extends PLS-DA by orthogonalizing the latent-variable decomposition to separate predictive variation (correlated with the class vector **y**) from orthogonal variation (uncorrelated with **y**). This simplifies score-plot interpretation and enhances biomarker identification. The `opls_da` class implements OPLS-DA via a scikit-learn Pipeline combining the `Scaler` preprocessing step, PLSRegression decomposition, and a custom `CrossValidation` estimator that computes correlation and covariance loading spectra, predictive scores (*t*_pred), orthogonal scores (*t*_ortho), and model fit metrics (R²X_corr, R²Y, Q²). Automatic component selection is available via the `auto_ncomp=True` flag.

The class provides a permutation test (`permutation_test()`) implementing the scikit-learn `permutation_test_score()` procedure to evaluate the null hypothesis that class separation is due to chance. VIP scores are computed from the PLS weight matrix according to the standard formula of Wold et al. (1993), with a sign transformation applied to indicate the direction of metabolite change between classes. Five visualization methods are available: `plot_oplsda_scores()` (scores scatter with confidence ellipses and R²X/R²Y/Q² annotations), `plot_hist()` (permutation score distribution with actual model accuracy), `plot_s_scores()` (S-plot of covariance vs. correlation), `plot_loading()` (pseudospectral loading colored by covariance), and `vip_plot()` (VIP scatter colored by direction and significance threshold).

#### 2.5.3 Lazy OPLS-DA

The `lazy_opls_da` class automates pairwise multi-group OPLS-DA by enumerating all unique group pairs from a metadata column, fitting an independent OPLS-DA model for each pair, and collecting results. This is particularly useful in studies with three or more experimental groups where exhaustive pairwise comparisons are required without manual sub-setting of datasets.

### 2.6 Univariate Statistics

The `UnivarStats` class in `utility.py` extends the multivariate analysis with per-feature univariate testing, including t-tests, Mann–Whitney U tests, and FDR correction via the Benjamini–Hochberg procedure (implemented through the pingouin library). The class is designed to complement OPLS-DA VIP analysis by providing statistical evidence for individual metabolite differences.

### 2.7 Interactive Applications

Two Dash-based web applications are bundled with metbit for users who prefer graphical exploration over programmatic interfaces. `STOCSY_app` launches a browser-based interactive STOCSY tool in which the anchor ppm value and p-value threshold can be adjusted with sliders and the correlation plot updates in real time. `pickie_peak` provides an interactive peak picker for NMR spectra with click-to-annotate functionality. Both applications run as local servers and require no external hosting infrastructure.

---

## 3. Results

### 3.1 Workflow Demonstration

To illustrate the integrated workflow, we describe a representative NMR metabolomics analysis performed using metbit. Beginning with a Bruker FID dataset, a researcher instantiates `nmr_preprocessing`, which reads the FID files with NMRglue, removes the digital filter, performs zero-filling and Fourier transformation, applies automatic phase correction, corrects baseline drift using the AsLS algorithm, and calibrates chemical shifts to the TSP reference signal. The resulting processed spectra are exported as a pandas DataFrame in which columns represent chemical shift positions (in ppm) and rows represent samples, providing a universal interchange format compatible with all downstream metbit modules.

Probabilistic quotient normalization is applied using `Normalization.pqn_normalization()` to account for sample dilution variation, followed by peak alignment with `icoshift_align()` to correct residual chemical-shift drift. A PCA model is then fitted using pareto scaling with `pca.fit()`, and the scores plot (`plot_pca_scores()`) is examined to identify outliers and assess overall sample clustering. Time-trajectory plots (`plot_pca_trajectory()`) visualize the temporal evolution of metabolic profiles across experimental time points.

Binary group comparisons proceed with `opls_da.fit()`, which fits the OPLS-DA model and reports R²X_corr, R²Y, and Q² values. Model validity is confirmed by `permutation_test()`, which generates the permutation score histogram (`plot_hist()`) showing that the observed classification accuracy exceeds the null distribution at *p* < 0.05. VIP scores are computed with `vip_scores()` and visualized with `vip_plot()` to rank metabolic features by their contribution to class discrimination, with sign encoding indicating the direction of change. The S-plot (`plot_s_scores()`) is examined to identify metabolites with both high covariance (quantitative contribution to the model) and high absolute correlation (reliability of the contribution). Finally, STOCSY analysis identifies resonance clusters that co-vary with key discriminatory signals, facilitating structural assignment of biomarker candidates.

### 3.2 Scalability and Performance

The package has been tested on datasets ranging from small pilot studies (n = 20 samples, ~30,000 spectral variables) to mid-scale studies (n ≥ 200 samples). OPLS-DA fitting for a 500-sample × 50,000-variable dataset completes in approximately 2.5 seconds on a standard multi-core workstation (as reported in the package documentation), with permutation tests parallelized using the `n_jobs=-1` argument via the scikit-learn joblib backend. PCA scaling and decomposition are performed using the highly optimized LAPACK routines in scikit-learn, ensuring O(min(n, p)²) complexity.

---

## 4. Discussion

### 4.1 Comparison with Existing Tools

metbit occupies a distinctive position among NMR metabolomics software tools. Commercial platforms such as Bruker TopSpin and Mnova offer comprehensive GUI-driven processing but lack scripting integration with the broader scientific Python ecosystem, and their licensing costs create barriers for academic and resource-limited settings. Web-based platforms such as MetaboAnalyst (Pang et al., 2022) and HMDB provide powerful analytical capabilities but require data upload to external servers and offer limited customization. R-based packages such as nmr, BATMAN, and ropls are well-validated but require proficiency in R and are not natively interoperable with Python deep-learning and machine-learning frameworks that are increasingly important in metabolomics research. nmrglue, the closest Python analogue, focuses on low-level spectral manipulation and does not include multivariate statistical modeling or interactive visualization.

In contrast, metbit provides the complete analytical pipeline within a single Python package installable via pip, enabling integration with Jupyter notebooks, reproducible workflow managers (Snakemake, Nextflow), and machine-learning pipelines based on scikit-learn, PyTorch, or TensorFlow. The use of Plotly for all visualizations ensures that all figures are natively interactive (hover, zoom, pan) in Jupyter notebook and HTML export contexts, without requiring additional JavaScript frameworks.

### 4.2 Limitations and Future Directions

Several limitations of the current implementation should be acknowledged. The preprocessing pipeline currently supports Bruker FID format exclusively; support for Varian/Agilent and JEOL formats is planned for future releases through extended NMRglue integration. The OPLS-DA implementation is restricted to binary classification; multi-class extension via hierarchical or multi-level OPLS-DA is under active development. The automated phase correction relies on the peak-minima algorithm, which may fail on spectra with broad baseline features or very low signal-to-noise ratios; manual override parameters are available as a workaround. Targeted metabolite quantification and automated spectral deconvolution are not currently implemented but represent high-priority development directions.

Future versions of metbit will incorporate support for 2D NMR spectroscopy (HSQC, COSY), expanded support for mass-spectrometry-based metabolomics datasets, integration with metabolite databases (HMDB, BMRB) for automated annotation, and deep-learning-based spectral classification models. The interactive Dash applications will be extended with additional analytical functions and will be made deployable as cloud-accessible web services.

### 4.3 Software Engineering and Reproducibility

metbit follows modern Python software engineering practices. The package is available on PyPI and GitHub under the MIT license, ensuring broad accessibility and permissive reuse. A full documentation website is hosted at https://metbit-docs.vercel.app and includes a getting-started guide, API reference, and annotated workflow examples. The repository includes a continuous integration test suite covering core preprocessing, scaling, normalization, and modeling functions, as well as security analysis via CodeQL and automated dependency management via Dependabot. The use of pandas DataFrames as the canonical data interchange format throughout the package ensures interoperability with the broader scientific Python ecosystem, including record linkage with clinical metadata, machine learning with scikit-learn, and statistical reporting with pingouin and statsmodels.

---

## 5. Conclusion

We present metbit, a comprehensive and freely available Python package that unifies the complete NMR-based metabolomics analytical workflow under a single, coherent API. By integrating NMR file reading, preprocessing, normalization, peak alignment, STOCSY, PCA, and OPLS-DA with production-quality interactive visualizations and Dash applications, metbit substantially lowers the technical barrier to reproducible NMR metabolomics analysis and facilitates integration with the modern scientific Python ecosystem. We invite the metabolomics community to adopt, contribute to, and extend metbit as a shared computational resource.

---

## Author Contributions

**Theerayut Bubpamala**: Conceptualization; Software (package design, implementation, and testing); Formal Analysis; Writing – Original Draft; Writing – Review & Editing; Visualization; Project Administration.


## Acknowledgements

The author thanks the open-source scientific Python community for the libraries on which metbit depends. The following generative AI tools provided auxiliary support during this project and are disclosed in accordance with ICMJE (2023) and Nature Portfolio AI authorship policies. These systems are not listed as authors because they cannot accept accountability for the work, cannot consent to authorship, and do not satisfy the criteria of intellectual contribution and approval of the final version required of human authors; the corresponding author (T.B.) accepts full responsibility for all content.

| Tool | Version | Role in this project |
|---|---|---|
| Theerayut Bubpamala | N/A | Developer of the Python package; manuscript author and primary writer |
| Claude | Sonnet 4.6 (Anthropic) | Manuscript drafting assistance and manuscript review |
| GPT | GPT-5.2 (OpenAI) | Code review; manuscript review |

## Funding and Conflicts of Interest

metbit is independently developed and maintained by kawa-technology. This work received no external funding. The author declares no conflicts of interest.

## Data Availability

metbit is freely available at https://github.com/aeiwz/metbit (MIT License). The package is installable via `pip install metbit`. Documentation is available at https://metbit-docs.vercel.app.

---

## References

Baek, S.-J., Park, A., Ahn, Y.-J., and Choo, J. (2015). Baseline correction using asymmetrically reweighted penalized least squares smoothing. *Analyst*, 140(1), 250-257.

van den Berg, R. A., Hoefsloot, H. C. J., Westerhuis, J. A., Smilde, A. K., and van der Werf, M. J. (2006). Centering, scaling, and transformations: improving the biological information content of metabolomics data. *BMC Genomics*, 7(1), 142.

Brand, A., Allen, L., Altman, M., Hlava, M., and Scott, J. (2015). Beyond authorship: attribution, contribution, collaboration, and credit. *Learned Publishing*, 28(2), 151-155. https://doi.org/10.1087/20150211

Chen, L., Weng, Z., Goh, L., and Garland, M. (2002). An efficient algorithm for automatic phase correction of NMR spectra based on entropy minimization. *Journal of Magnetic Resonance*, 158(1-2), 164-168.

da Costa-Luis, C., Larroque, S. K., Altendorf, K., Mary, H., richardsheridan, Korobov, M., Yorav-Raphael, N., Ivanov, I., Bargull, M., Rodrigues, N., Chen, G., Lee, A., Newey, C., CrazyPython, and contributors. (2023). tqdm: A fast, Extensible Progress Bar for Python and CLI. *Zenodo*. https://doi.org/10.5281/zenodo.8233024

Dieterle, F., Ross, A., Schlotterbeck, G., and Senn, H. (2006). Probabilistic quotient normalization as a robust method to account for dilution of complex biological mixtures. Application in 1H NMR metabonomics. *Analytical Chemistry*, 78(13), 4281-4290.

Eilers, P. H. C., and Boelens, H. F. M. (2005). Baseline Correction with Asymmetric Least Squares Smoothing. Leiden University Medical Centre Report.

Emwas, A.-H., Roy, R., McKay, R. T., Tenori, L., Saccenti, E., Gowda, G. A. N., Raftery, D., Alahmari, F., Jaremko, L., Jaremko, M., and Wishart, D. S. (2019). NMR spectroscopy for metabolomics research. *Metabolites*, 9(7), 123.

Erb, A. (2022). pybaselines: A Python library of algorithms for the baseline correction of experimental data. *Journal of Open Source Software*, 7(78), 4554. https://doi.org/10.21105/joss.04554

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Rio, J. F., Wiebe, M., Peterson, P., Gerard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C., and Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362. https://doi.org/10.1038/s41586-020-2649-2

Helmus, J. J., and Jaroniec, C. P. (2013). Nmrglue: an open source Python package for the analysis of multidimensional NMR data. *Journal of Biomolecular NMR*, 55(4), 355-367. https://doi.org/10.1007/s10858-013-9718-x

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science and Engineering*, 9(3), 90-95. https://doi.org/10.1109/MCSE.2007.55

Martens, H., and Stark, E. (1991). Extended multiplicative signal correction and spectral interference subtraction: new preprocessing methods for near infrared spectroscopy. *Journal of Pharmaceutical and Biomedical Analysis*, 9(8), 625-635.

McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt and J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, pp. 56-61. https://doi.org/10.25080/Majora-92bf1922-00a

Nicholson, J. K., Lindon, J. C., and Holmes, E. (1999). Metabonomics: understanding the metabolic responses of living systems to pathophysiological stimuli via multivariate statistical analysis of biological NMR spectroscopic data. *Xenobiotica*, 29(11), 1181-1189.

Nicholson, J. K., Foxall, P. J., Spraul, M., Farrant, R. D., and Lindon, J. C. (2005). 750 MHz 1H and 1H-13C NMR spectroscopy of human blood plasma. *Analytical Chemistry*, 77(19), 6283-6293.

The pandas Development Team. (2020). *pandas-dev/pandas: Pandas*. Zenodo. https://doi.org/10.5281/zenodo.3509134

Pang, Z., Chong, J., Zhou, G., de Lima Morais, D. A., Chang, L., Barrette, M., Gauthier, C., Jacques, P.-E., Li, S., and Xia, J. (2022). MetaboAnalyst 5.0: narrowing the gap between raw spectra and functional insights. *Nucleic Acids Research*, 50(W1), W537-W544.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Plotly Technologies Inc. (2015). *Collaborative data science*. Plotly Technologies Inc., Montreal, QC. https://plot.ly

Plotly Technologies Inc. (2017). *Dash: Analytical web applications for Python, R, Julia, and Jupyter (no JavaScript required)*. Plotly Technologies Inc., Montreal, QC. https://dash.plotly.com

Savorani, F., Tomasi, G., and Engelsen, S. B. (2010). icoshift: A versatile tool for the rapid alignment of 1D NMR spectra. *Journal of Magnetic Resonance*, 202(2), 190-202.

Seabold, S., and Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In S. van der Walt and J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, pp. 57-61. https://doi.org/10.25080/Majora-92bf1922-011

Trygg, J., and Wold, S. (2002). Orthogonal projections to latent structures (O-PLS). *Journal of Chemometrics*, 16(3), 119-128.

Vallat, R. (2018). Pingouin: statistics in Python. *Journal of Open Source Software*, 3(31), 1026. https://doi.org/10.21105/joss.01026

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., Carey, C. J., Polat, I., Feng, Y., Moore, E. W., VanderPlas, J., Laxalde, D., Perktold, J., Cimrman, R., Henriksen, I., Quintero, E. A., Harris, C. R., Archibald, A. M., Ribeiro, A. H., Pedregosa, F., van Mulbregt, P., and SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17(3), 261-272. https://doi.org/10.1038/s41592-019-0686-2

Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021. https://doi.org/10.21105/joss.03021

Wishart, D. S., Guo, A., Oler, E., Wang, F., Anjum, A., Peters, H., Dizon, R., Sayeeda, Z., Tian, S., Lee, B. L., Berjanskii, M., Mah, R., Yamamoto, M., Jovel, J., Torres-Calzada, C., Hiebert-Lauderdale, M., Pon, A., Budinski, Z., Chin, J., Bertozzi, S. M., Lau, J. X., Nickel, J., Sokolenko, S., Li, H., Motlagh, J., Tymensen, L., and Srivastava, P. (2022). HMDB 5.0: the Human Metabolome Database for 2022. *Nucleic Acids Research*, 50(D1), D622-D631.

Wold, S., Johansson, E., and Cocchi, M. (1993). PLS: Partial Least Squares Projections to Latent Structures. In H. Kubinyi (Ed.), *3D QSAR in Drug Design*, pp. 523-550. ESCOM, Leiden.

Zhang, Z.-M., Chen, S., and Liang, Y.-Z. (2010). Baseline correction using adaptive iteratively reweighted penalized least squares. *Analyst*, 135(5), 1138-1146.
