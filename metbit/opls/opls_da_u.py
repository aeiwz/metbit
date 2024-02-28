
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import cross_validation
import plotting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

from pyChemometrics import ChemometricsScaler

import os
from lingress.unipair import Unipair




#Make directory
# path folder

data = pd.read_csv('/Volumes/CAS9/Aeiwz/Documents/Thesis/Analyse/Dataset/U_noesy_pqn.csv')

#exclude INT_H in index number 59
data = data.drop([59])


pair = Unipair(meta=data, column_name='Class')

dataset = pair.get_dataset()

names = pair.get_name()


path_ = "/Volumes/CAS9/Aeiwz/Documents/Thesis/Analyse/Report/OPLSDA_U"
os.makedirs(path_, exist_ok=True)

# Create directories if they don't exist
os.makedirs('{}/element'.format(path_), exist_ok=True)
os.makedirs('{}/element/hist_plot'.format(path_), exist_ok=True)
os.makedirs('{}/element/loading_plot'.format(path_), exist_ok=True)
os.makedirs('{}/element/score_plot'.format(path_), exist_ok=True)
os.makedirs('{}/element/s_plot'.format(path_), exist_ok=True)
os.makedirs('{}/main'.format(path_), exist_ok=True)
os.makedirs('{}/element/Lingress'.format(path_), exist_ok=True)

# Import the datasets from the /data directory


# X for the NMR spectra and Y for the 2 outcome variables
path_save = "{}/element".format(path_)
#test group
df = data
#Drop QC samples
#df = df.drop(df[df['Technique'] == 'QC'].index)

Group="Class"



for i in range(len(dataset)):
    

    plot_name = names[i]


    test_select = dataset[i]

    
    
    X = test_select.iloc[:, 14:]
    df_X = X
    #fill nan with 0
    X = X.fillna(0)
    meta = test_select.iloc[:, :14]
    Y = test_select[Group]
    Y1 = pd.Categorical(Y).codes
    ppm = list(np.ravel(X.columns).astype(float))
    # Use pandas Categorical type to generate the dummy enconding of the Y vector (0 and 1) 
    name = names[i]
   

    spectra = test_select.iloc[:, 14:]
    ppm = list(spectra.columns.astype(float))
    X = spectra.values
    y = Y
    
    # Create a pipeline with data preprocessing and OPLS-DA model
    pipeline = Pipeline([
                            ('scale', ChemometricsScaler(scale_power=0.5)),
                            ('oplsda', PLSRegression(n_components=2)),
                            ('opls', cross_validation.CrossValidation(kfold=3, estimator='opls', scaler='pareto'))
                         ])

    oplsda = pipeline.named_steps['oplsda']
    cv = pipeline.named_steps['opls']
    cv.fit(X, y)

    oplsda.fit(X, pd.Categorical(y).codes)
    n_permutate = 1000

    # Permutation test to assess the significance of the model
    acc_score, permutation_scores, p_value = permutation_test_score(
    pipeline.named_steps['oplsda'], X, pd.Categorical(y).codes, cv=3, n_permutations=n_permutate, n_jobs=-1, random_state=57, verbose=10)


    s_scores_df = pd.DataFrame({'correlation': cv.correlation,'covariance': cv.covariance}, index=ppm)
    df_opls_scores = pd.DataFrame({'t_scores': cv.scores, 't_ortho': cv.orthogonal_score, 't_pred': cv.predictive_score, 'label': y})

        
    colour_dict = {
                    "sham ad libitum": "#F55D4D",        
                    "CR + INT777 (H)": "#58E6BE",
                    "CR + INT777 (L)": "#4E8BF5",       
                    }
    
    symbol_dict = {'1-wk pre-op': 'circle',
                    '1-wk post-op': 'diamond',
                    '2-wk post-op': 'square',
                    '4-wk post-op': 'triangle-up'
                    }


    #Visualise
    from pca_ellipse import confidence_ellipse
    fig = px.scatter(df_opls_scores, x='t_scores', y='t_ortho', symbol=meta['Time point'], 
                
                    symbol_map=symbol_dict,
                
                    color='label', 
                    color_discrete_map=colour_dict, 
                    title='<b>OPLS-DA Scores Plot<b>', 
                    height=900, width=1300,
                    labels={
                        't_pred': 't<sub>predict</sub>',
                        't_ortho': 't<sub>orthogonal</sub>',
                        't_scores': 't<sub>scores</sub>',
                        'label': 'Intervention'}
                    )

    #fig.add_annotation(yref = 'paper', y = -1.06, xref = 'paper', x=1.06 , text='Q2' +' = {}'.format(np.round(df_explained_variance_.iloc[2,2], decimals=2)))
    #fig.update_annotations(font = {
    #    'size': 20}, showarrow=False)

    #set data point fill alpha with boarder in each color
    fig.update_traces(marker=dict(size=35, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')))

    fig.add_annotation(dict(font=dict(color="black",size=20),
                            #x=x_loc,
                            x=0,
                            y=1.04+0.05,
                            showarrow=False,
                            text='<b>R<sup>2</sup>X: {}%<b>'.format(np.round(cv.R2Xcorr*100, decimals=2)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")

    fig.add_annotation(dict(font=dict(color="black",size=20),
                            #x=x_loc,
                            x=0,
                            y=1.0+0.05,
                            showarrow=False,
                            text='<b>R<sup>2</sup>Y: {}%<b>'.format(np.round(cv.R2y*100, decimals=2)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")
    fig.add_annotation(dict(font=dict(color="black",size=20),
                            #x=x_loc,
                            x=0,
                            y=1.08+0.05,
                            showarrow=False,
                            text='<b>Q<sup>2</sup>: {}%<b>'.format(np.round(cv.q2*100, decimals=2)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")

    fig.add_shape(type='path',
            path=confidence_ellipse(df_opls_scores['t_scores'], df_opls_scores['t_ortho']))


    fig.update_traces(marker=dict(size=35))
    #fig.update_traces(textposition='top center') #Text label position
    #change M to 10^6
    fig.update_yaxes(tickformat=",.0")
    fig.update_xaxes(tickformat=",.0")

    #fig.update_traces(marker=dict(size=12, color=Y1_color, marker=Y2_marker))

    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(
        title={
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(size=20))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    #fig.show()
    fig.write_image("{}/score_plot/score_plot_{}.png".format(path_save, name))
    fig.write_html("{}/score_plot/score_plot{}.html".format(path_save, name))




    #Histrogram
    #Plot histogram of permutation scores
    fig = px.histogram(permutation_scores, nbins=50, height=500, width=1000, 
                    title='<b>Permutation scores<b>',
                    labels={'value': 'Accuracy score', 
                            'count': 'Frequency'})
    #add dashed line to indicate the accuracy score of the model line y location is maximum count of histogram
    fig.add_shape(type='line', yref='paper', y0=0, y1=1, xref='x', x0=acc_score, x1=acc_score, line=dict(dash='dash', color='red', width=3))


    fig.add_annotation(dict(font=dict(color="black",size=14),
                            #x=x_loc,
                            x=0,
                            y=1.25,
                            #y=1.18,
                            showarrow=False,
                            text='Number of permutation: {}'.format(n_permutate),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            #x=x_loc,
                            x=0,
                            y=1.18,
                            showarrow=False,
                            text='Accuracy score: {}'.format(np.round(acc_score, decimals=3)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            #x=x_loc,
                            x=0,
                            y=1.11,
                            showarrow=False,
                            text='<i>p-value</i>: {}'.format(np.round(p_value, decimals=6)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")

    fig.update_layout(showlegend=False)

    fig.update_layout(title_x=0.5)

    #fig.show()
    fig.write_image("{}/hist_plot/Permutation_scores_{}.png".format(path_save, name))
    fig.write_html("{}/hist_plot/Permutation_scores_{}.html".format(path_save, name))
    
    
    
    #S plot
    # sub-plot covariance for x and correlation for y S-plot using plotly, color by covariance with jet colormap
    #setup figure size


    fig = px.scatter(s_scores_df, x='covariance', y='correlation', color='covariance', range_color=[-1,1],
                     color_continuous_scale='jet', text=s_scores_df.index, height=900, width=2000)
    fig.update_layout(title='<b>S-plot</b>', xaxis_title='Covariance', yaxis_title='Correlation')

    #add line of axis and set color to black and line width to 2 pixel
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    #Add tick width to 2 pixel
    fig.update_xaxes(tickwidth=2)
    fig.update_yaxes(tickwidth=2)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(tickformat=",.0")
    #fig.update_xaxes(tickformat=",.0")
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(
        title={
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(size=20))
    #Set font size to 20
    #Set marker size to 5 pixel
    fig.update_traces(marker=dict(size=14))
    #fig.show()
    fig.write_image("{}/s_plot/S_plot_{}.png".format(path_save, name))
    fig.write_html("{}/s_plot/S_plot_{}.html".format(path_save, name))
    

    #Loadings plot
    
    # X * 1 when correlation is positive, X * -1 when correlation is negative
    def median_corr(X):
        X_corr = np.median(X, axis=0)
        X_corr = X_corr * np.sign(s_scores_df['correlation'])
        return X_corr

    X2 = median_corr(X)

    fig = px.scatter(s_scores_df, x=ppm, y=X2, color='covariance', color_continuous_scale='jet', text=s_scores_df.index, height=500, width=2000)

    fig.update_traces(marker=dict(size=3))
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(title='<b>Median spectra</b>', xaxis_title='ppm', yaxis_title='Correlation')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    #Add tick width to 2 pixel
    fig.update_xaxes(tickwidth=2)
    fig.update_yaxes(tickwidth=2)

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(tickformat=",.0")
    #fig.update_xaxes(tickformat=",.0")
    fig.update_layout(
        title={
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(size=20))
    #Set marker size to 5 pixel
    fig.update_traces(marker=dict(size=3))
    #fig.show()
    fig.write_image("{}/loading_plot/loadings_plot_{}.png".format(path_save, name))
    fig.write_html("{}/loading_plot/loadings_plot_{}.html".format(path_save, name))
    
    from lingress.Lingress import lin_regression
    lin_mod = lin_regression(x=df_X, target=meta[Group], label=meta[Group], features_name=ppm)
    lin_mod.create_dataset()
    lin_mod.fit_model(adj_method='fdr_bh')
    report = lin_mod.report()
    report.to_csv("{}/Lingress/lingress_report_{}.csv".format(path_save, name))
    lin_mod.volcano_plot()
    lin_mod.png_plot(plot_name="lingress/volcano_plot_{}".format(name), path_save=path_save)
    lin_mod.html_plot(plot_name="lingress/volcano_plot_{}".format(name), path_save=path_save)

    del X, Y, Y1, ppm, spectra, meta, df_X, s_scores_df, df_opls_scores, cv, oplsda, pipeline, acc_score, permutation_scores, p_value, fig, lin_mod, report, n_permutate, plot_name, 
    
    print("Finish {}".format(name))

