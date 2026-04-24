# Manuscript Review and Journal Recommendations

## Manuscript Review: metbit
**Title:** metbit: An Integrated Python Package for End-to-End NMR-Based Metabolomics Data Analysis  
**Reviewer:** Gemini CLI  
**Date:** 2026-04-24

### Summary
The manuscript provides a comprehensive overview of the `metbit` package (v8.7.7), highlighting its role in consolidating the fragmented NMR metabolomics workflow into a single Pythonic pipeline. The package's integration of raw Bruker data processing, spectral normalization (PQN, MSC), peak alignment (icoshift), and multivariate statistics (PCA, OPLS-DA) is a significant contribution to the field.

### Strengths
- **End-to-End Integration:** Successfully bridges the gap between raw spectral data and biological interpretation within a single Python library.
- **Reproducibility:** Emphasis on scriptable workflows and scikit-learn compatible interfaces supports the "Open Science" mandate.
- **Interactive Visualization:** The use of Plotly and Dash for interactive figures and apps (STOCSY, peak picking) enhances the user experience for non-programmers.
- **Robust Methodology:** Implementation of established algorithms (PQN, AsLS, icoshift) ensures scientific validity.

### Recommendations
- The manuscript is ready for submission to software-focused or metabolomics journals.
- Consider highlighting the scalability to large cohorts (n > 200) as a key selling point in the abstract.

---

## Journal Recommendations (No-Fee / Diamond OA)

The following journals offer publication options with **no Article Processing Charges (APCs)**, either through a Diamond Open Access model or a traditional subscription route (where the article is free for authors but behind a paywall for readers).

| Journal Name | Model | Focus | Why it fits? |
| --- | --- | --- | --- |
| **Journal of Open Source Software (JOSS)** | Diamond OA | Research Software | Perfect fit for `metbit` as it focuses on the software itself and has no fees. |
| **Bioinformatics (Oxford)** | Hybrid (Free Subscription) | Bioinformatics Tools | High-impact journal with a dedicated "Software" section. |
| **Metabolomics (Springer)** | Hybrid (Free Subscription) | Metabolomics | Official journal of the Metabolomics Society; very relevant audience. |
| **Journal of Proteome Research (ACS)** | Hybrid (Free Subscription) | Proteomics/Metabolomics | Frequently publishes NMR-based metabolomics studies and methods. |
| **Computers in Biology and Medicine (Elsevier)** | Hybrid (Free Subscription) | Computational Biology | Good for tools bridging computational methods and biological applications. |
| **Analytica Chimica Acta** | Hybrid (Free Subscription) | Analytical Chemistry | Appropriate for the chemical-shift calibration and normalization methodologies. |

**Note:** For hybrid journals, ensure you select the **"Subscription"** (non-open access) option to avoid the APC. If you choose "Open Access," you will likely be charged a fee.
