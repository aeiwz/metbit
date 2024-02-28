# _*_ coding: utf-8 _*_

import numpy as np
import pandas as pd

__author__ = "aeiwz"

 
class vip_scores:
    
    def __init__(self, model, features_name = None):
        self.model = model
        self.features_name = features_name
        
        features_name = self.features_name
        model = self.model


        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
       
        if features_name is not None:
            vips = pd.DataFrame(vips, columns = ['VIP'])
            vips['Features'] = features_name
        else:
            vips = pd.DataFrame(vips, columns = ['VIP'])
            vips['Features'] = vips.index

            
        self.vips = vips

        return

    def get_scores(self):
        vips = self.vips
        return vips

	
    def vip_plot(self, threshold = 2):
        # add scatter plot of VIP score
        import plotly.express as px
        vips = self.vips

        fig = px.scatter(vips, x='Features', y='VIP', text='Features', color='VIP', color_continuous_scale='jet', range_color=(0, 2.5), height=500, width=1000, title='VIP score')
        fig.update_traces(marker=dict(size=12))
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig.update_yaxes(tickformat=",.00")
        fig.update_xaxes(tickformat=",.00")
        fig.update_layout(
            title={
                'y':1,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            font=dict(size=20))

        # reverse the x-axis
        fig.update_xaxes(autorange="reversed")
        
        # add dashed line for threshold
        fig.add_shape(type="line",
                    x0=0, y0=threshold, x1=10, y1=threshold,
                    line=dict(color="red",width=2, dash="dash"))
                    
        fig.update_layout(showlegend=False)
        
        return fig



if __name__ == '__main__':
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    model = PLSRegression(n_components=3)
    model.fit(X, y)
    vip = vip_scores(model, diabetes.feature_names)
    print(vip.vip())
    
    model = PLSRegression(n_components=3)
    model.fit(X, y)
    vip = vip_scores(model)
    print(vip.vip())
