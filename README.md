# metbit

Metbit is a Python package designed for the analysis of metabolomics data. It provides a range of tools and functions to process, visualize, and interpret metabolomics datasets. With Metbit, you can perform various statistical analyses, identify biomarkers, and generate informative visualizations to gain insights into your metabolomics experiments. Whether you are a researcher, scientist, or data analyst working in the field of metabolomics, Metbit can help streamline your data analysis workflow and facilitate the interpretation of complex metabolomics data.


# How to install

```bash
pip install metbit
```

# Example:
---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.3
  nbformat: 4
  nbformat_minor: 2
---

::: {.cell .markdown}
# **Principal component analysis**

PCA is used to transform a large set of variables into a smaller one
that still contains most of the information in the large set. This is
particularly useful when dealing with high-dimensional data, where
visualizing and analyzing the data can be challenging.

To perform Principal Component Analysis (PCA) in a Python environment,
you can use the metbit library. Here\'s a step-by-step guide to import
the necessary package and perform PCA:
:::

::: {.cell .code execution_count="2" trusted="true"}
``` python
import pandas as pd
from metbit import pca
```
:::

::: {.cell .code execution_count="3" trusted="true"}
``` python
df = pd.read_csv("metbit_tutorial_data.csv")
```
:::

::: {.cell .code execution_count="30"}
``` python
df.iloc[:10, :10].to_markdown('test.md', index=False)
```
:::

::: {.cell .code execution_count="4" trusted="true"}
``` python
X = df.iloc[:, 2:]
ppm = X.columns.astype(float).to_list()
color_ = df["Group"]
symbol_ = df["Time point"]
time_order = {1:0, 2:1, 3:2, 4:2}
```
:::

:::: {.cell .code execution_count="5" trusted="true"}
``` python
pca_mod = pca(X=X, label=color_, features_name=ppm, n_components=3)
pca_mod.fit()
```

::: {.output .execute_result execution_count="5"}
<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA(n_components=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;PCA<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.decomposition.PCA.html">?<span>Documentation for PCA</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>PCA(n_components=3)</pre></div> </div></div></div></div>
:::
::::

::: {.cell .code execution_count="35" trusted="true"}
``` python
pca_mod.plot_pca_scores(pc=["PC1", "PC2"], symbol_=symbol_).write_image("pca_scores[PC1-PC2].svg")
```
:::

::: {.cell .code execution_count="43" trusted="true"}
``` python
pca_mod.plot_pca_scores(pc=["PC1", "PC3"], symbol_=symbol_).write_image("pca_scores[PC1-PC3].svg")
```
:::

::: {.cell .code execution_count="42" trusted="true"}
``` python
pca_mod.plot_3d_pca(marker_size=10, symbol_=symbol_).write_image("3d_pca.svg")
```
:::

::: {.cell .code execution_count="41" trusted="true"}
``` python
pca_mod.plot_pca_trajectory(time_=symbol_, time_order=time_order, pc=["PC1", "PC2"]).write_image("pca_trajectory[PC1-PC2].svg")
```
:::

::: {.cell .code execution_count="40" trusted="true"}
``` python
pca_mod.plot_pca_trajectory(time_=symbol_, time_order=time_order, pc=["PC1", "PC3"]).write_image("pca_trajectory[PC1-PC3].svg")
```
:::

::: {.cell .markdown}
# **Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA)**

Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA) was
proposed by Prof. Svante Wold in 2002 as a variant of PLS-DA, using a
mathematical filter to remove systematic variance unrelated to the
sample class. This is particularly advantageous in metabolomics, such as
distinguishing the metabolomic signature of coronary disease without
confounding factors like sex. However, OPLS-DA is less common than
PLS-DA due to increased risk of overfitting and its limitation to binary
classification.
:::

::: {.cell .code execution_count="17"}
``` python

from metbit import opls_da 
import pandas as pd 
```
:::

::: {.cell .markdown}
1.  Load the data and data manipulation
:::

::: {.cell .code execution_count="18" trusted="true"}
``` python
df = pd.read_csv("metbit_tutorial_data.csv")
#Exclude base line (Time point 1)
df.drop(df.loc[df["Time point"]==1].index, inplace=True)
```
:::

::: {.cell .code execution_count="19" trusted="true"}
``` python
X = df.iloc[:, 2:]
ppm = X.columns.astype(float).to_list()
y = df["Group"]
```
:::

::: {.cell .code execution_count="20" trusted="true"}
``` python
opls_da_mod = opls_da(X=X, y=y, features_name=ppm, auto_ncomp=True)
```
:::

:::: {.cell .code execution_count="21" trusted="true"}
``` python
opls_da_mod.fit()
```

::: {.output .stream .stdout}
    OPLS-DA model is fitted in 26.940966844558716 seconds
:::
::::

::: {.cell .code execution_count="44" trusted="true"}
``` python
opls_da_mod.plot_oplsda_scores().write_image("oplsda_scores.svg")
```
:::

::: {.cell .code execution_count="45" trusted="true"}
``` python
opls_da_mod.plot_loading().write_image("oplsda_loading.svg")
```
:::

::: {.cell .code execution_count="46" trusted="true"}
``` python
opls_da_mod.plot_s_scores().write_image("oplsda_s_scores.svg")
```
:::

:::::: {.cell .code execution_count="25" trusted="true"}
``` python
opls_da_mod.permutation_test(n_permutations=100, n_jobs=-1)
```

::: {.output .stream .stderr}
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    8.5s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   11.1s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   13.2s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:   17.5s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   20.8s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:   23.7s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:   27.4s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:   33.5s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:   37.9s
    [Parallel(n_jobs=-1)]: Done  96 out of 100 | elapsed:   42.7s remaining:    1.8s
:::

::: {.output .stream .stdout}
    Permutation test is performed in 46.19982290267944 seconds
:::

::: {.output .stream .stderr}
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   43.6s finished
:::
::::::

::: {.cell .code execution_count="47" trusted="true"}
``` python
opls_da_mod.plot_hist().write_image("oplsda_hist.svg")
```
:::

::: {.cell .code execution_count="48" trusted="true"}
``` python
opls_da_mod.vip_scores()
opls_da_mod.vip_plot(threshold=2).write_image("oplsda_vip_plot.svg")
```
:::

::: {.cell .markdown}
# **Additional**

# **Lazy OPLS-DA**
:::

::: {.cell .code execution_count="38"}
``` python
import pandas as pd
from metbit import lazy_opls_da
```
:::

::: {.cell .code execution_count="39"}
``` python
df = pd.read_csv("metbit_tutorial_data.csv")
```
:::

::: {.cell .code execution_count="40"}
``` python
X = df.iloc[:, 2:]
ppm = X.columns.astype(float).to_list()
# Perform class by combind Group and Time point
df["Class"] = df["Group"] + ", " + df["Time point"].astype(str)
y = df["Class"]
```
:::

:::: {.cell .code execution_count="48"}
``` python
working_dir = "/path/to/working/directory/"
lazy_mod = lazy_opls_da(data=X, groups=y,working_dir=working_dir, auto_ncomp=True, permutation=True, VIP=True, linear_regression=True)
```

::: {.output .stream .stdout}

            Project Name: 2024-06-26 13:34:49_PulsePioneer
            Number of groups: 8
            Number of samples: 81
            Number of features: 58262
            Number of components: 2
            Estimator: opls
            Scaling: pareto
            Kfold: 3
            Random state: 94
            Auto ncomp: True
            Working directory: /Volumes/CAS9/Aeiwz/test flight/metbit tutorial
            Permutation: True
            VIP: True
            Linear regression: True
            
:::
::::

::::::::::: {.cell .code execution_count="49"}
``` python
lazy_mod.fit()
```

::: {.output .stream .stdout}
    OPLS-DA model is fitted in 2.101414203643799 seconds
:::

::: {.output .stream .stderr}
    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:    6.7s
    [Parallel(n_jobs=4)]: Done  10 tasks      | elapsed:    9.1s
    [Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:   12.9s
    [Parallel(n_jobs=4)]: Done  24 tasks      | elapsed:   15.9s
    [Parallel(n_jobs=4)]: Done  33 tasks      | elapsed:   20.7s
    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:   24.2s
    [Parallel(n_jobs=4)]: Done  53 tasks      | elapsed:   28.4s
    [Parallel(n_jobs=4)]: Done  64 tasks      | elapsed:   32.4s
    [Parallel(n_jobs=4)]: Done  77 tasks      | elapsed:   37.6s
    [Parallel(n_jobs=4)]: Done  90 tasks      | elapsed:   42.1s
    [Parallel(n_jobs=4)]: Done 100 out of 100 | elapsed:   45.9s finished
:::

::: {.output .stream .stdout}
    Permutation test is performed in 48.356366872787476 seconds
:::

::: {.output .stream .stderr}
    Creating data frame: 100%|██████████| 58262/58262 [00:00<00:00, 1792662.19it/s]
    Features processed: 58262it [01:22, 707.95it/s]                    
:::

::: {.output .stream .stdout}
    adjustment p-value with Benjamini/Hochberg (non-negative) Done
    OPLS-DA model is fitted in 1.6124029159545898 seconds
:::

::: {.output .stream .stderr}
    [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:    3.3s
    [Parallel(n_jobs=4)]: Done  10 tasks      | elapsed:    4.9s
    [Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:    7.8s
    [Parallel(n_jobs=4)]: Done  24 tasks      | elapsed:   10.0s
    [Parallel(n_jobs=4)]: Done  33 tasks      | elapsed:   13.8s
    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:   17.5s
    [Parallel(n_jobs=4)]: Done  53 tasks      | elapsed:   21.6s
    [Parallel(n_jobs=4)]: Done  64 tasks      | elapsed:   25.6s
    [Parallel(n_jobs=4)]: Done  77 tasks      | elapsed:   30.2s
    [Parallel(n_jobs=4)]: Done  90 tasks      | elapsed:   35.0s
    [Parallel(n_jobs=4)]: Done 100 out of 100 | elapsed:   38.6s finished
:::

::: {.output .stream .stdout}
    Permutation test is performed in 40.054511070251465 seconds
:::

::: {.output .stream .stderr}
    Creating data frame: 100%|██████████| 58262/58262 [00:00<00:00, 3837626.45it/s]
    Features processed: 15662it [00:21, 732.89it/s]           Exception ignored in: <bound method IPythonKernel._clean_thread_parent_frames of <ipykernel.ipkernel.IPythonKernel object at 0x10b29fe00>>
    Traceback (most recent call last):
      File "/opt/homebrew/lib/python3.12/site-packages/ipykernel/ipkernel.py", line 775, in _clean_thread_parent_frames
        def _clean_thread_parent_frames(

    KeyboardInterrupt: 
    Features processed: 24431it [00:34, 726.25it/s]
:::
:::::::::::
