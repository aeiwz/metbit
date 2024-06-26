# metbit

Metbit is a Python package designed for the analysis of metabolomics data. It provides a range of tools and functions to process, visualize, and interpret metabolomics datasets. With Metbit, you can perform various statistical analyses, identify biomarkers, and generate informative visualizations to gain insights into your metabolomics experiments. Whether you are a researcher, scientist, or data analyst working in the field of metabolomics, Metbit can help streamline your data analysis workflow and facilitate the interpretation of complex metabolomics data.


# How to install

```bash
pip install metbit
```

# Example:
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Principal component analysis**\n",
    "\n",
    "PCA is used to transform a large set of variables into a smaller one that still contains most of the information in the large set. This is particularly useful when dealing with high-dimensional data, where visualizing and analyzing the data can be challenging.\n",
    "\n",
    "\n",
    "To perform Principal Component Analysis (PCA) in a Python environment, you can use the metbit library. Here's a step-by-step guide to import the necessary package and perform PCA:\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from metbit import pca"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"metbit_tutorial_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.iloc[:10, :10].to_markdown('test.md', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "X = df.iloc[:, 2:]\n",
    "ppm = X.columns.astype(float).to_list()\n",
    "color_ = df[\"Group\"]\n",
    "symbol_ = df[\"Time point\"]\n",
    "time_order = {1:0, 2:1, 3:2, 4:2}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "trusted": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {\n",
       "  /* Definition of color scheme common for light and dark mode */\n",
       "  --sklearn-color-text: black;\n",
       "  --sklearn-color-line: gray;\n",
       "  /* Definition of color scheme for unfitted estimators */\n",
       "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
       "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
       "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
       "  --sklearn-color-unfitted-level-3: chocolate;\n",
       "  /* Definition of color scheme for fitted estimators */\n",
       "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
       "  --sklearn-color-fitted-level-1: #d4ebff;\n",
       "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
       "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
       "\n",
       "  /* Specific color for light theme */\n",
       "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
       "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
       "  --sklearn-color-icon: #696969;\n",
       "\n",
       "  @media (prefers-color-scheme: dark) {\n",
       "    /* Redefinition of color scheme for dark theme */\n",
       "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
       "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
       "    --sklearn-color-icon: #878787;\n",
       "  }\n",
       "}\n",
       "\n",
       "#sk-container-id-1 {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 pre {\n",
       "  padding: 0;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-hidden--visually {\n",
       "  border: 0;\n",
       "  clip: rect(1px 1px 1px 1px);\n",
       "  clip: rect(1px, 1px, 1px, 1px);\n",
       "  height: 1px;\n",
       "  margin: -1px;\n",
       "  overflow: hidden;\n",
       "  padding: 0;\n",
       "  position: absolute;\n",
       "  width: 1px;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-dashed-wrapped {\n",
       "  border: 1px dashed var(--sklearn-color-line);\n",
       "  margin: 0 0.4em 0.5em 0.4em;\n",
       "  box-sizing: border-box;\n",
       "  padding-bottom: 0.4em;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-container {\n",
       "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
       "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
       "     so we also need the `!important` here to be able to override the\n",
       "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
       "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
       "  display: inline-block !important;\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-text-repr-fallback {\n",
       "  display: none;\n",
       "}\n",
       "\n",
       "div.sk-parallel-item,\n",
       "div.sk-serial,\n",
       "div.sk-item {\n",
       "  /* draw centered vertical line to link estimators */\n",
       "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
       "  background-size: 2px 100%;\n",
       "  background-repeat: no-repeat;\n",
       "  background-position: center center;\n",
       "}\n",
       "\n",
       "/* Parallel-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item::after {\n",
       "  content: \"\";\n",
       "  width: 100%;\n",
       "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
       "  flex-grow: 1;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel {\n",
       "  display: flex;\n",
       "  align-items: stretch;\n",
       "  justify-content: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  position: relative;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
       "  align-self: flex-end;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
       "  align-self: flex-start;\n",
       "  width: 50%;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
       "  width: 0;\n",
       "}\n",
       "\n",
       "/* Serial-specific style estimator block */\n",
       "\n",
       "#sk-container-id-1 div.sk-serial {\n",
       "  display: flex;\n",
       "  flex-direction: column;\n",
       "  align-items: center;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  padding-right: 1em;\n",
       "  padding-left: 1em;\n",
       "}\n",
       "\n",
       "\n",
       "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
       "clickable and can be expanded/collapsed.\n",
       "- Pipeline and ColumnTransformer use this feature and define the default style\n",
       "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
       "*/\n",
       "\n",
       "/* Pipeline and ColumnTransformer style (default) */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable {\n",
       "  /* Default theme specific background. It is overwritten whether we have a\n",
       "  specific estimator or a Pipeline/ColumnTransformer */\n",
       "  background-color: var(--sklearn-color-background);\n",
       "}\n",
       "\n",
       "/* Toggleable label */\n",
       "#sk-container-id-1 label.sk-toggleable__label {\n",
       "  cursor: pointer;\n",
       "  display: block;\n",
       "  width: 100%;\n",
       "  margin-bottom: 0;\n",
       "  padding: 0.5em;\n",
       "  box-sizing: border-box;\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
       "  /* Arrow on the left of the label */\n",
       "  content: \"▸\";\n",
       "  float: left;\n",
       "  margin-right: 0.25em;\n",
       "  color: var(--sklearn-color-icon);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
       "  color: var(--sklearn-color-text);\n",
       "}\n",
       "\n",
       "/* Toggleable content - dropdown */\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content {\n",
       "  max-height: 0;\n",
       "  max-width: 0;\n",
       "  overflow: hidden;\n",
       "  text-align: left;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content pre {\n",
       "  margin: 0.2em;\n",
       "  border-radius: 0.25em;\n",
       "  color: var(--sklearn-color-text);\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
       "  /* Expand drop-down */\n",
       "  max-height: 200px;\n",
       "  max-width: 100%;\n",
       "  overflow: auto;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
       "  content: \"▾\";\n",
       "}\n",
       "\n",
       "/* Pipeline/ColumnTransformer-specific style */\n",
       "\n",
       "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator-specific style */\n",
       "\n",
       "/* Colorize estimator box */\n",
       "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  /* The background is the default theme color */\n",
       "  color: var(--sklearn-color-text-on-default-background);\n",
       "}\n",
       "\n",
       "/* On hover, darken the color of the background */\n",
       "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "/* Label box, darken color on hover, fitted */\n",
       "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
       "  color: var(--sklearn-color-text);\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Estimator label */\n",
       "\n",
       "#sk-container-id-1 div.sk-label label {\n",
       "  font-family: monospace;\n",
       "  font-weight: bold;\n",
       "  display: inline-block;\n",
       "  line-height: 1.2em;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-label-container {\n",
       "  text-align: center;\n",
       "}\n",
       "\n",
       "/* Estimator-specific */\n",
       "#sk-container-id-1 div.sk-estimator {\n",
       "  font-family: monospace;\n",
       "  border: 1px dotted var(--sklearn-color-border-box);\n",
       "  border-radius: 0.25em;\n",
       "  box-sizing: border-box;\n",
       "  margin-bottom: 0.5em;\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-0);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-0);\n",
       "}\n",
       "\n",
       "/* on hover */\n",
       "#sk-container-id-1 div.sk-estimator:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-2);\n",
       "}\n",
       "\n",
       "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-2);\n",
       "}\n",
       "\n",
       "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
       "\n",
       "/* Common style for \"i\" and \"?\" */\n",
       "\n",
       ".sk-estimator-doc-link,\n",
       "a:link.sk-estimator-doc-link,\n",
       "a:visited.sk-estimator-doc-link {\n",
       "  float: right;\n",
       "  font-size: smaller;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1em;\n",
       "  height: 1em;\n",
       "  width: 1em;\n",
       "  text-decoration: none !important;\n",
       "  margin-left: 1ex;\n",
       "  /* unfitted */\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted,\n",
       "a:link.sk-estimator-doc-link.fitted,\n",
       "a:visited.sk-estimator-doc-link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
       ".sk-estimator-doc-link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover,\n",
       "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
       ".sk-estimator-doc-link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "/* Span, style for the box shown on hovering the info icon */\n",
       ".sk-estimator-doc-link span {\n",
       "  display: none;\n",
       "  z-index: 9999;\n",
       "  position: relative;\n",
       "  font-weight: normal;\n",
       "  right: .2ex;\n",
       "  padding: .5ex;\n",
       "  margin: .5ex;\n",
       "  width: min-content;\n",
       "  min-width: 20ex;\n",
       "  max-width: 50ex;\n",
       "  color: var(--sklearn-color-text);\n",
       "  box-shadow: 2pt 2pt 4pt #999;\n",
       "  /* unfitted */\n",
       "  background: var(--sklearn-color-unfitted-level-0);\n",
       "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link.fitted span {\n",
       "  /* fitted */\n",
       "  background: var(--sklearn-color-fitted-level-0);\n",
       "  border: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "\n",
       ".sk-estimator-doc-link:hover span {\n",
       "  display: block;\n",
       "}\n",
       "\n",
       "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link {\n",
       "  float: right;\n",
       "  font-size: 1rem;\n",
       "  line-height: 1em;\n",
       "  font-family: monospace;\n",
       "  background-color: var(--sklearn-color-background);\n",
       "  border-radius: 1rem;\n",
       "  height: 1rem;\n",
       "  width: 1rem;\n",
       "  text-decoration: none;\n",
       "  /* unfitted */\n",
       "  color: var(--sklearn-color-unfitted-level-1);\n",
       "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
       "  /* fitted */\n",
       "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
       "  color: var(--sklearn-color-fitted-level-1);\n",
       "}\n",
       "\n",
       "/* On hover */\n",
       "#sk-container-id-1 a.estimator_doc_link:hover {\n",
       "  /* unfitted */\n",
       "  background-color: var(--sklearn-color-unfitted-level-3);\n",
       "  color: var(--sklearn-color-background);\n",
       "  text-decoration: none;\n",
       "}\n",
       "\n",
       "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
       "  /* fitted */\n",
       "  background-color: var(--sklearn-color-fitted-level-3);\n",
       "}\n",
       "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>PCA(n_components=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow fitted\">&nbsp;&nbsp;PCA<a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.4/modules/generated/sklearn.decomposition.PCA.html\">?<span>Documentation for PCA</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></label><div class=\"sk-toggleable__content fitted\"><pre>PCA(n_components=3)</pre></div> </div></div></div></div>"
      ],
      "text/plain": [
       "PCA(n_components=3)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pca_mod = pca(X=X, label=color_, features_name=ppm, n_components=3)\n",
    "pca_mod.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "pca_mod.plot_pca_scores(pc=[\"PC1\", \"PC2\"], symbol_=symbol_).write_image(\"pca_scores[PC1-PC2].svg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "pca_mod.plot_pca_scores(pc=[\"PC1\", \"PC3\"], symbol_=symbol_).write_image(\"pca_scores[PC1-PC3].svg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "pca_mod.plot_3d_pca(marker_size=10, symbol_=symbol_).write_image(\"3d_pca.svg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "pca_mod.plot_pca_trajectory(time_=symbol_, time_order=time_order, pc=[\"PC1\", \"PC2\"]).write_image(\"pca_trajectory[PC1-PC2].svg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "pca_mod.plot_pca_trajectory(time_=symbol_, time_order=time_order, pc=[\"PC1\", \"PC3\"]).write_image(\"pca_trajectory[PC1-PC3].svg\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA)**\n",
    "\n",
    "Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA) was proposed by Prof. Svante Wold in 2002 as a variant of PLS-DA, using a mathematical filter to remove systematic variance unrelated to the sample class. This is particularly advantageous in metabolomics, such as distinguishing the metabolomic signature of coronary disease without confounding factors like sex. However, OPLS-DA is less common than PLS-DA due to increased risk of overfitting and its limitation to binary classification."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "from metbit import opls_da \n",
    "import pandas as pd "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. Load the data and data manipulation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"metbit_tutorial_data.csv\")\n",
    "#Exclude base line (Time point 1)\n",
    "df.drop(df.loc[df[\"Time point\"]==1].index, inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "X = df.iloc[:, 2:]\n",
    "ppm = X.columns.astype(float).to_list()\n",
    "y = df[\"Group\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "opls_da_mod = opls_da(X=X, y=y, features_name=ppm, auto_ncomp=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "trusted": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "OPLS-DA model is fitted in 26.940966844558716 seconds\n"
     ]
    }
   ],
   "source": [
    "opls_da_mod.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "opls_da_mod.plot_oplsda_scores().write_image(\"oplsda_scores.svg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "opls_da_mod.plot_loading().write_image(\"oplsda_loading.svg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "opls_da_mod.plot_s_scores().write_image(\"oplsda_s_scores.svg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "trusted": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.\n",
      "[Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    8.5s\n",
      "[Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   11.1s\n",
      "[Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   13.2s\n",
      "[Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:   17.5s\n",
      "[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   20.8s\n",
      "[Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:   23.7s\n",
      "[Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:   27.4s\n",
      "[Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:   33.5s\n",
      "[Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:   37.9s\n",
      "[Parallel(n_jobs=-1)]: Done  96 out of 100 | elapsed:   42.7s remaining:    1.8s\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Permutation test is performed in 46.19982290267944 seconds\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   43.6s finished\n"
     ]
    }
   ],
   "source": [
    "opls_da_mod.permutation_test(n_permutations=100, n_jobs=-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "opls_da_mod.plot_hist().write_image(\"oplsda_hist.svg\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "opls_da_mod.vip_scores()\n",
    "opls_da_mod.vip_plot(threshold=2).write_image(\"oplsda_vip_plot.svg\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Additional**\n",
    "\n",
    "# **Lazy OPLS-DA**\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from metbit import lazy_opls_da"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"metbit_tutorial_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.iloc[:, 2:]\n",
    "ppm = X.columns.astype(float).to_list()\n",
    "# Perform class by combind Group and Time point\n",
    "df[\"Class\"] = df[\"Group\"] + \", \" + df[\"Time point\"].astype(str)\n",
    "y = df[\"Class\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "        Project Name: 2024-06-26 13:34:49_PulsePioneer\n",
      "        Number of groups: 8\n",
      "        Number of samples: 81\n",
      "        Number of features: 58262\n",
      "        Number of components: 2\n",
      "        Estimator: opls\n",
      "        Scaling: pareto\n",
      "        Kfold: 3\n",
      "        Random state: 94\n",
      "        Auto ncomp: True\n",
      "        Working directory: /Volumes/CAS9/Aeiwz/test flight/metbit tutorial\n",
      "        Permutation: True\n",
      "        VIP: True\n",
      "        Linear regression: True\n",
      "        \n"
     ]
    }
   ],
   "source": [
    "working_dir = \"/path/to/working/directory/\"\n",
    "lazy_mod = lazy_opls_da(data=X, groups=y,working_dir=working_dir, auto_ncomp=True, permutation=True, VIP=True, linear_regression=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "OPLS-DA model is fitted in 2.101414203643799 seconds\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.\n",
      "[Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:    6.7s\n",
      "[Parallel(n_jobs=4)]: Done  10 tasks      | elapsed:    9.1s\n",
      "[Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:   12.9s\n",
      "[Parallel(n_jobs=4)]: Done  24 tasks      | elapsed:   15.9s\n",
      "[Parallel(n_jobs=4)]: Done  33 tasks      | elapsed:   20.7s\n",
      "[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:   24.2s\n",
      "[Parallel(n_jobs=4)]: Done  53 tasks      | elapsed:   28.4s\n",
      "[Parallel(n_jobs=4)]: Done  64 tasks      | elapsed:   32.4s\n",
      "[Parallel(n_jobs=4)]: Done  77 tasks      | elapsed:   37.6s\n",
      "[Parallel(n_jobs=4)]: Done  90 tasks      | elapsed:   42.1s\n",
      "[Parallel(n_jobs=4)]: Done 100 out of 100 | elapsed:   45.9s finished\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Permutation test is performed in 48.356366872787476 seconds\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Creating data frame: 100%|██████████| 58262/58262 [00:00<00:00, 1792662.19it/s]\n",
      "Features processed: 58262it [01:22, 707.95it/s]                    \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "adjustment p-value with Benjamini/Hochberg (non-negative) Done\n",
      "OPLS-DA model is fitted in 1.6124029159545898 seconds\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.\n",
      "[Parallel(n_jobs=4)]: Done   5 tasks      | elapsed:    3.3s\n",
      "[Parallel(n_jobs=4)]: Done  10 tasks      | elapsed:    4.9s\n",
      "[Parallel(n_jobs=4)]: Done  17 tasks      | elapsed:    7.8s\n",
      "[Parallel(n_jobs=4)]: Done  24 tasks      | elapsed:   10.0s\n",
      "[Parallel(n_jobs=4)]: Done  33 tasks      | elapsed:   13.8s\n",
      "[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:   17.5s\n",
      "[Parallel(n_jobs=4)]: Done  53 tasks      | elapsed:   21.6s\n",
      "[Parallel(n_jobs=4)]: Done  64 tasks      | elapsed:   25.6s\n",
      "[Parallel(n_jobs=4)]: Done  77 tasks      | elapsed:   30.2s\n",
      "[Parallel(n_jobs=4)]: Done  90 tasks      | elapsed:   35.0s\n",
      "[Parallel(n_jobs=4)]: Done 100 out of 100 | elapsed:   38.6s finished\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Permutation test is performed in 40.054511070251465 seconds\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Creating data frame: 100%|██████████| 58262/58262 [00:00<00:00, 3837626.45it/s]\n",
      "Features processed: 15662it [00:21, 732.89it/s]           Exception ignored in: <bound method IPythonKernel._clean_thread_parent_frames of <ipykernel.ipkernel.IPythonKernel object at 0x10b29fe00>>\n",
      "Traceback (most recent call last):\n",
      "  File \"/opt/homebrew/lib/python3.12/site-packages/ipykernel/ipkernel.py\", line 775, in _clean_thread_parent_frames\n",
      "    def _clean_thread_parent_frames(\n",
      "\n",
      "KeyboardInterrupt: \n",
      "Features processed: 24431it [00:34, 726.25it/s]"
     ]
    }
   ],
   "source": [
    "lazy_mod.fit()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

