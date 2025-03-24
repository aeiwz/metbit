# -*- coding: utf-8 -*-

__auther__ ='aeiwz'
author_email='theerayut_aeiw_123@hotmail.com'
__copyright__="Copyright 2024, Theerayut"

__license__ = "MIT"
__maintainer__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__status__ = "Develop"
class lazy_opls_da:

    '''
    Parameters:
    -----------
        •	data (pd.DataFrame): DataFrame containing the dataset.
        •	groups (list): List of class labels for each data sample.
        •	working_dir (str): Directory path for storing output files.
        •	feature_names (list, optional): Names of features, defaults to None.
        •	n_components (int, optional): Number of components for OPLS-DA, defaults to 2.
        •	scaling (str, optional): Scaling method ('pareto'), defaults to 'pareto'.
        •	estimator (str, optional): Model estimator, defaults to 'opls'.
        •	kfold (int, optional): Number of folds in cross-validation, defaults to 3.
        •	random_state (int, optional): Random seed, defaults to 94.
        •	auto_ncomp (bool, optional): Automatically choose the optimal number of components, defaults to True.
        •	permutation (bool, optional): Conduct permutation tests, defaults to True.
        •	VIP (bool, optional): Calculate VIP scores, defaults to True.
        •	linear_regression (bool, optional): Conduct linear regression analysis, defaults to True.

    Returns:
    --------
        •	A printout of the model summary, including the project name, dataset information, configuration, and directory paths.

    fit Method

    Fits the OPLS-DA model to the dataset, generates plots, and saves them to the output directory.

    Parameters:
    -----------
        •	marker_color (dict, optional): Dictionary mapping groups to colors.
        •	custom_color (list, optional): Custom color grouping.
        •	custom_shape (list, optional): Custom shape grouping.
        •	symbol_dict (dict, optional): Dictionary mapping groups to marker symbols.
        •	custom_legend_name (list, optional): Custom names for the legend, defaults to ['Group', 'Sub-group'].
        •	marker_label (str or None, optional): Specifies marker labels ('class', 'group', or 'sub-group').
        •	marker_size (int or None, optional): Size of markers in plots.
        •	marker_opacity (float or None, optional): Opacity level of markers in plots.
        •	individual_ellipse (bool, optional): Option to display individual ellipses for each group.

    Returns:
    --------
        •	A message indicating the model fitting was successful.

    Directory and Project Setup
    ===========================
    Creates necessary folders in the working directory based on project needs (e.g., for VIP score plots, permutation scores, etc.). Paths are stored in a dictionary (self.path).

    Directories Created:
    --------------------
        •	working_dir/project_name/element/plots/... for different plots.
        •	working_dir/project_name/element/data/... for data outputs.

    Plotting and Saving Data
    ------------------------
        1.	Score Plot: Generates OPLS-DA score plots for each group.
        2.	Loading Plot: Generates and saves loading plots.
        3.	S Plot: Generates and saves S-score plots.
        4.	VIP Score Plot: Generates VIP score plots and saves VIP scores as CSV if VIP=True.
        5.	Permutation Test Plot: Conducts permutation tests and saves permutation scores as CSV if permutation=True.
        6.	Volcano Plot (Linear Regression): Generates volcano plot and saves data if linear_regression=True.
    '''

    import os
    from glob import glob
    import pandas as pd
    import numpy as np
    import random

    if __name__ == '__main__':
        from metbit import opls_da
        from metbit.utility import project_name_generator
        from metbit.utility import lazypair
    else:
        from .metbit import opls_da
        from .utility import project_name_generator
        from .utility import lazypair
    

    def __init__(self, data: pd.DataFrame, groups: list, working_dir: str, feature_names: list = None, n_components: int = 2, scaling: str = 'pareto', 
                    estimator: str = 'opls', kfold: int = 3, random_state: int = 94, auto_ncomp: bool = True,  
                    permutation: bool = True, n_permutation: int = 500, n_jobs: int = 4,
                    VIP: bool = True, VIP_threshold: float = 1.5, 
                    linear_regression: bool = True, FC_threshold: float = 1.5, p_val_threshold: float = 2) -> None:

        import os
        from glob import glob
        import pandas as pd
        import numpy as np
        import random
        from .metbit import opls_da

        from .utility import project_name_generator
        from .utility import lazypair

        """
        This function takes in a dataframe and a list of y values and returns the project_name model.
        Parameters
        ----------
        data: pandas dataframe
            The dataframe to be used.
        y: list
            The list of y values.

        n_components: int
            The number of components to use.
        lazy_opls_da(data, y, n_components).fit()

        """


        self.groups = groups
        self.n_components = n_components
        self.working_dir = working_dir
        self.feature_names = feature_names

        self.random_state = random_state        
        self.estimator = estimator
        self.scale = scaling
        self.kfold = kfold
        self.auto_ncomp = auto_ncomp
        

        data['Class'] = groups
        self.data = data

        self.permutation = permutation
        if permutation == True:
            self.n_permutation = n_components
            self.n_jobs = n_jobs
        else:
            pass
        
        self.VIP = VIP
        if VIP == True:
            self.VIP_threshold = VIP_threshold
        else:
            pass

        self.linear_regression = linear_regression
        if linear_regression == True:
            self.FC_threshold = FC_threshold
            self.p_val_threshold = p_val_threshold
        else:
            pass

        """
        This function takes in a dataframe and a list of y values and returns the project_name model.
        Parameters
        ----------
        data: pandas dataframe
            The dataframe to be used.
        y: list
            The list of y values.
        n_components: int
            The number of components to use.
        lazy_opls_da(data, y, n_components).fit()

        """
        
        project_name = project_name_generator()
        

        #Remove last / from working_dir
        if working_dir[-1] == '/':
            working_dir = working_dir[:-1]
        else:
            working_dir = working_dir

        #Replace \ with / for windows
        working_dir = working_dir.replace('\\', '/')


        if os.path.exists(working_dir + '/' + project_name + '/element'):
            print('Directory already exist')
        else:
            folder_name_plot = ['loading_plot', 's_plot', 'score_plot']

            folder_name_data = ['Score_data', 'Loading_data']

            if permutation == True:
                folder_name_plot.append('hist_plot')
                folder_name_data.append('Permutation_scores')
            else:
                pass
            if VIP == True:
                folder_name_plot.append('VIP_score_plot')
                folder_name_data.append('VIP_scores')
            else:
                pass
            if linear_regression == True:
                folder_name_plot.append('Volcano_plot')
                folder_name_data.append('Lingress_data')
            else:
                pass



            os.makedirs(working_dir+'/' + project_name + '/element', exist_ok=True)
            for i in folder_name_plot:
                os.makedirs(working_dir+'/' + project_name + '/element/plots/' + i, exist_ok=True)
            for i in folder_name_data:
                os.makedirs(working_dir+'/' + project_name + '/element/data/' + i, exist_ok=True)

            os.makedirs(working_dir+'/' + project_name + '/main')

        #Create dictionary to store the path
        dir = glob(working_dir + '/' + project_name + '/element/*/*/')
        dir = [path_.replace('\\', '/') for path_ in dir]

        #Create dictionary to store the path
        path = {}
        for i in dir:
            path[i.split('/')[-2]] = i


        self.path = path

        #Print summary model as table text format
        Summary = f"""
        Project Name: {project_name}
        Number of groups: {len(data['Class'].unique())}
        Number of samples: {len(data)}
        Number of features: {len(data.columns) - 1}
        Number of components: {n_components}
        Estimator: {estimator}
        Scaling: {scaling}
        Kfold: {kfold}
        Random state: {random_state}
        Auto ncomp: {auto_ncomp}
        Working directory: {working_dir}
        Output directory: {working_dir}/{project_name}
        Permutation: {permutation}
        Number of permutation: {self.n_permutation if permutation else 'None'}
        VIP: {VIP}
        VIP threshold: {self.VIP_threshold if VIP else 'None'}
        Linear regression: {linear_regression}
        Fold change threshold: {self.FC_threshold if linear_regression else 'None'}
        P-value threshold: {self.p_val_threshold if linear_regression else 'None'}
        """


        return print(Summary)



    def fit(self, marker_color: dict = None, custom_color: list = None, 
            custom_shape: list = None, symbol_dict: dict = None, 
            custom_legend_name = ['Group', 'Sub-group'], 
            marker_label=None, marker_size=None, marker_opacity=None, 
            individual_ellipse=False) -> None:


        from .metbit import opls_da
        from lingress import lin_regression
        from .utility import lazypair

        from glob import glob
        import pandas as pd
        import numpy as np
        import random
        import plotly

     
        data = self.data
        n_components = self.n_components
        path = self.path
        scale = self.scale
        feature_names = self.feature_names

        from plotly.validators.scatter.marker import SymbolValidator
        raw_symbols = SymbolValidator().values
        namestems = []
        for i in range(0,len(raw_symbols),3):
            name = raw_symbols[i+2]
            namestems.append(name)

        if marker_color is None:

            import plotly.colors as plotly_colour
            name_color_set = ['Plotly', 'D3', 'G10', 'T10', 'Alphabet', 'Dark24', 'Light24', 'Set1', 'Pastel1', 
                        'Dark2', 'Set2', 'Pastel2', 'Set3', 'Antique', 'Safe', 'Bold', 'Pastel', 
                        'Vivid', 'Prism']

            palette = []
            for name in name_color_set:
                palette += getattr(plotly_colour.qualitative, name) # This is a list of colors

            marker_color = {}
            unique_groups = data['Class'].unique()

            # Make sure the palette has enough colors for the unique groups
            if len(unique_groups) > len(palette):
                raise ValueError("The palette does not have enough colors for all unique groups")

            # Map each unique group to a color
            for group, color in zip(unique_groups, palette):
                marker_color[group] = color
        else:
            marker_color = marker_color

        if custom_color is None:
            data['Group'] = data['Class']
        else:
            data['Group'] = custom_color

        if custom_shape is None:
            data['Sub-group'] = data['Class']
        else:
            data['Sub-group'] = custom_shape

        if symbol_dict is None:
            symbol_dict = {}
            for i in data['Class'].unique():
                #random symbol from plotly symbol
                symbol_dict[i] = random.choice(namestems)
        else:
            symbol_dict = symbol_dict


        #Create object attribute
        lazy = lazypair(data, 'Class')
        data_list = lazy.get_dataset()
        name_save = lazy.get_name()

        for i in range(len(data_list)):
            
            df = data_list[i]
            name = name_save[i]

            meta = df.loc[:, ['Class', 'Group', 'Sub-group']]
            X = df.drop(['Class', 'Group', 'Sub-group'], axis=1)
            if feature_names is None:
                try:
                    feature_names = X.columns.astype(float).to_list()
                except:
                    feature_names = X.columns.to_list()
            else: 
                feature_names = feature_names
            y = meta['Class']


            if marker_label is None:
                marker_label_2 = marker_label
            
            else:
                if marker_label == 'class':
                    marker_label_2 = meta['class']
                elif marker_label == 'group':
                    marker_label_2 = meta['Group']
                elif marker_label == 'sub-group':
                    marker_label_2 = meta['Sub-group']
                elif marker_label == 'index':
                    marker_label_2 = meta.index
                else:
                    marker_label_2 = meta.index


            #OPLS-DA
            oplsda_mod = opls_da(X=X, y=y, features_name = feature_names, n_components=n_components, scaling_method=scale, 
                                    estimator=self.estimator, kfold=self.kfold, random_state=self.random_state, 
                                    auto_ncomp = self.auto_ncomp)
            oplsda_mod.fit()

            if marker_size is not None:
                marker_size = marker_size
            else:
                if len(df) <= 100:
                    marker_score_size = 35
                else:
                    marker_score_size = 16

            if marker_opacity is not None:
                marker_opacity = marker_opacity
            else:
                marker_opacity = 0.7
                
            #Score plot
            if custom_shape is not None:
                oplsda_mod.plot_oplsda_scores(color_=meta['Group'], color_dict=marker_color, symbol_=meta['Sub-group'], symbol_dict=symbol_dict, marker_size=marker_score_size, marker_opacity=marker_opacity, marker_label=marker_label_2, 
                legend_name=custom_legend_name, individual_ellipse=individual_ellipse).write_html(path['score_plot'] + name + '_score_plot.html')
                oplsda_mod.plot_oplsda_scores(color_=meta['Group'], color_dict=marker_color, symbol_=meta['Sub-group'], symbol_dict=symbol_dict, marker_size=marker_score_size, marker_opacity=marker_opacity, marker_label=marker_label_2, 
                legend_name=custom_legend_name, individual_ellipse=individual_ellipse).write_image(path['score_plot'] + name + '_score_plot.png')
            else:
                oplsda_mod.plot_oplsda_scores(color_dict=marker_color, marker_size=marker_score_size, marker_opacity=marker_opacity, marker_label=marker_label_2, 
                legend_name=custom_legend_name, individual_ellipse=individual_ellipse).write_html(path['score_plot'] + name + '_score_plot.html')
                oplsda_mod.plot_oplsda_scores(color_dict=marker_color, marker_size=marker_score_size, marker_opacity=marker_opacity, marker_label=marker_label_2, 
                legend_name=custom_legend_name, individual_ellipse=individual_ellipse).write_image(path['score_plot'] + name + '_score_plot.png')


            oplsda_mod.get_oplsda_scores().to_csv(path['Score_data'] + name + '_score_data.csv', index=False)
            oplsda_mod.get_s_scores().to_csv(path['Loading_data'] + name + '_loading_data.csv', index=False)

            if len(feature_names) >= 100:
                marker_load_size = 5
            else:
                marker_load_size = 20
            #Loading plot
            oplsda_mod.plot_loading(xaxis_title='Features', marker_size=marker_load_size).write_html(path['loading_plot'] + name + '_loading_plot.html')
            oplsda_mod.plot_loading(xaxis_title='Features', marker_size=marker_load_size).write_image(path['loading_plot'] + name + '_loading_plot.png')

            #S plot
            oplsda_mod.plot_s_scores(marker_size=marker_load_size).write_html(path['s_plot'] + name + '_s_plot.html')
            oplsda_mod.plot_s_scores(marker_size=marker_load_size).write_image(path['s_plot'] + name + '_s_plot.png')

            #VIP score plot
            if self.VIP == True:
                oplsda_mod.vip_scores()
                oplsda_mod.get_vip_scores().to_csv(path['VIP_scores'] + name + '_VIP_scores.csv')
                oplsda_mod.vip_plot(threshold=self.VIP_threshold, marker_size=marker_load_size).write_html(path['VIP_score_plot'] + name + '_VIP_score_plot.html')
                oplsda_mod.vip_plot(threshold=self.VIP_threshold, marker_size=marker_load_size).write_image(path['VIP_score_plot'] + name + '_VIP_score_plot.png')
            else:
                pass

            #Permutation test
            if self.permutation == True:
                oplsda_mod.permutation_test(n_permutations=self.n_permutation, n_jobs=self.n_jobs)
                oplsda_mod.plot_hist().write_html(path['hist_plot'] + name + '_hist_plot.html')
                oplsda_mod.plot_hist().write_image(path['hist_plot'] + name + '_hist_plot.png')
                permutation_score_np = oplsda_mod.get_permutation_scores()
                permutation_score_df = pd.DataFrame(permutation_score_np, columns=['Permutation_scores'])
                permutation_score_df.to_csv(path['Permutation_scores'] + name + '_permutation_scores.csv', index=False)
            else:
                pass

            #Linear regression
            if self.linear_regression == True:
                lin_ = lin_regression(x=X, target=y, label=y, features_name=feature_names)
                lin_.create_dataset()
                lin_.fit_model(adj_method='fdr_bh')
                lin_.volcano_plot(fc_cut_off=self.FC_threshold, p_val_cut_off=self.p_val_threshold).write_html(path['Volcano_plot'] + name + '_Volcano_plot.html')
                lin_.volcano_plot(fc_cut_off=self.FC_threshold, p_val_cut_off=self.p_val_threshold).write_image(path['Volcano_plot'] + name + '_Volcano_plot.png')
                lin_.report().to_csv(path['Lingress_data'] + name + '_Lingress_data.csv', index=False)
            else:
                pass
        
        return print('Model has been fitted successfully')

