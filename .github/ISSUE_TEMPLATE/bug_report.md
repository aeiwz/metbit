---
name: Bug report
about: Something broken in metbit - preprocessing, analysis, visualization, or data loading
title: '[Bug] '
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what went wrong.

**Affected area**
<!-- Check all that apply -->
- [ ] Data loading (Bruker reader, file import)
- [ ] Preprocessing (baseline, normalisation, scaling, alignment)
- [ ] Multivariate analysis (PCA, OPLS-DA)
- [ ] Statistics (fold change, p-value, VIP)
- [ ] Visualisation / Dash app
- [ ] Other (describe below)

**Minimal reproducible example**
```python
import metbit

# paste the smallest code that triggers the bug
```

**Error / traceback**
```
Paste the full traceback here
```

**Expected behaviour**
What should have happened instead?

**Environment**
- metbit version: <!-- `import metbit; print(metbit.__version__)` -->
- Python version: <!-- `python --version` -->
- OS: <!-- e.g. macOS 14, Ubuntu 22.04, Windows 11 -->
- Install method: <!-- pip / conda / from source -->

**Data context** *(optional)*
- Instrument / pulse sequence: 
- Spectrum type: <!-- 1D ¹H, TOCSY, etc. -->
- Approximate number of samples / variables:

**Additional context**
Any other information, screenshots of plots, or links to related issues.
