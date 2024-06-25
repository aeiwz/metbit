# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

__author__ = "aeiwz"

class lazypair:

    def __init__(self, dataset, column_name):
        
        meta = dataset
        self.meta = meta
        self.column_name = column_name
        

        """
        This function takes in a dataframe and a column name and returns the index of the dataframe and the names of the pairs
        of the unique values in the column.
        Parameters
        ----------
        meta: pandas dataframe
            The dataframe to be used.
        column_name: str
        Unipair(meta, column_name).indexing()
        
        """
        import pandas as pd
        import numpy as np
        
        #check unique values in the column
        if meta[column_name].nunique() < 3:
            raise ValueError("Group should contain at least 3 groups")
        else:
            pass
        #check meta is a dataframe
        if not isinstance(meta, pd.DataFrame):
            raise ValueError("meta should be a pandas dataframe")
        #check column_name is a string
        if not isinstance(column_name, str):
            raise ValueError("column_name should be a string")
        

        df = meta
        y = df[column_name].unique()
        pairs = []
        for i in range(len(y)):
            for j in range(i+1, len(y)):
                pairs.append([y[i], y[j]])
        
        index_ = []
        for i in range(len(pairs)):
            inside_index = []
            for j in range(2):
                inside_index.append(list((df.loc[df[column_name] == pairs[i][j]]).index))
            index_list = [inside_index[0] + inside_index[1]]
            index_.append(index_list[0])
        pairs
        index_
        names = []
        for i in range(len(pairs)):
            
            names.append(str(pairs[i][0]) + "_vs_" + str(pairs[i][1]))
            #check names if contain / replace with _ 
            names[i] = names[i].replace('/', '_')
            
        del df
        del y
        
        self.index_ = index_
        self.names = names
        
        
        

    def get_index(self):
        index_ = self.index_
        return index_
    
    def get_name(self):
        names = self.names
        return names
    
    def get_meta(self):
        meta = self.meta
        column_name = self.column_name
        return meta[column_name]
    
    def get_column_name(self):
        column_name = self.column_name
        return column_name
    
    def get_dataset(self):
        df = self.meta
        index_ = self.index_
        list_of_df = []
        for i in range(len(index_)):
            list_of_df.append(df.loc[index_[i]])
        
        #Create object attribute
        self.list_of_df = list_of_df
        return list_of_df

class gen_page:





    import sys
    import os
    import re
    import shutil
    from glob import glob
    import pandas as pd




    def __init__(self, data_path):
        '''
        This function takes in the path to the data folder and returns the HTML files for the OPLS-DA plots.
        Parameters
        ----------
        data_path: str
            The path to the data folder.
        gen_page(data_path).get_files()
        '''
        self.data_path = data_path

        if data_path[-1] == '/':
            #remove the last /
            data_path = data_path[:-1]
            
        else:
            data_path = data_path


        #check data_path is a string
        if not isinstance(data_path, str):
            raise ValueError("data_path should be a string")
        
        #check data_path is a directory
        if not os.path.isdir(data_path):
            raise ValueError("data_path should be a directory")

        #check if data_path is empty
        if not os.listdir(data_path):
            raise ValueError("data_path should not be empty")

        #check if data_path contains the necessary files
        if not os.path.exists(data_path+'/element/hist_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'hist_plot'")
        if not os.path.exists(data_path+'/element/Lingress'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'Lingress'")
        if not os.path.exists(data_path+'/element/loading_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'loading_plot'")
        if not os.path.exists(data_path+'/element/s_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 's_plot'")
        if not os.path.exists(data_path+'/element/score_plot'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'score_plot'")
        if not os.path.exists(data_path+'/element/VIP_score'):
            raise ValueError("data_path should contain a folder named 'element' with a folder named 'VIP_score'")



        #change directory to data_path
        os.chdir(data_path)

    def get_files(self):
        
        data_path = self.data_path


        hist_plot = glob(pathname= data_path+'/element/hist_plot/*.html')
        Lingress_ = glob(pathname= data_path+'/element/Lingress/*.html')
        loading_plot = glob(pathname= data_path+'/element/loading_plot/*.html')
        s_plot = glob(pathname= data_path+'/element/s_plot/*.html')
        score_plot = glob(pathname= data_path+'/element/score_plot/*.html')
        VIP_score = glob(pathname= data_path+'/element/VIP_score/*.html')

        files = pd.DataFrame({'hist_plot': hist_plot, 'Lingress': Lingress_, 'loading_plot': loading_plot, 's_plot': s_plot,'VIP': VIP_score, 'score_plot': score_plot})

        # Get the name of the files
        files['names'] = files['hist_plot'].str.split('/').str[-1].str.split('.').str[0]
        files['names'] = files['names'].str.replace('Permutation_scores_','')
        files['names'] = files['names'].str.replace(' ','_')
        #replace value in dataframe with ..
        files = files.replace(to_replace=data_path, value='..', regex=True)

        html_content_list = []

        # Iterate through the name and file_ lists to create HTML files
        for i in range(len(files)):
            html_content = f"""

        <!DOCTYPE html>
        <html lang="en">
        <head>
        <title>OPLS-DA</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="description" content="HTML5 website template">
        <meta name="keywords" content="CASIX, template, html, sass, jquery">
        <meta name="author" content="CASIX">

        <link rel="ix-icon" type="image/png" href="assets/img/logo.PNG">
        <link rel="shortcut icon" type="image/png" href="assets/img/logo.PNG">
        </head>
        <body>

        <body>

            <div class="container">
                <iframe src="{files['score_plot'][i]}" 
                frameborder="0" 
                width="100%" 
                height="1000"></iframe>
            </div>
            
            <div class="container">
                <iframe src="{files['loading_plot'][i]}" 
                frameborder="0" 
                width="100%" 
                height="1000"></iframe>
            </div>
            
            <div class="container">
                <iframe src="{files['s_plot'][i]}" 
                frameborder="0" 
                width="100%" 
                height="600"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['hist_plot'][i]}" 
                frameborder="0" 
                width="100%" height="600"></iframe>
            </div>

            
            <div class="container">
                <iframe src="{files['VIP'][i]}" 
                frameborder="0" 
                width="100%" height="600"></iframe>
            </div>

            <div class="container">
                <iframe src="{files['Lingress'][i]}" 
                frameborder="0" 
                width="100%" height="600"></iframe>
            </div>


        </body>
        </html>

            
            """
            html_content_list.append(html_content)
        
        for i in range(len(files)):
            # Write the HTML content to the file
            file_path = f"./main/oplsda_{files['names'][i]}.html"
            with open(file_path, "w") as html_file:
                html_file.write(html_content_list[i])

        return print('HTML files created')

            
class oplsda_path:

    def __init__(self, data_path):

        import os
        from glob import glob

        '''
        This function takes in the path to the data folder and returns the path to the OPLS-DA plots.
        Parameters
        ----------
        data_path: str
            The path to the data folder.
        oplsda_path(data_path).get_path()
        '''


        #check data_path is a string
        if not isinstance(data_path, str):
            raise ValueError("data_path should be a string")

        #check data_path is a directory
        if not os.path.isdir(data_path):
            raise ValueError("data_path should be a directory")

        #check if data_path is empty
        if not os.listdir(data_path):
            raise ValueError("data_path should not be empty")


        #Implement \ to / for windows
        data_path = data_path.replace('\\', '/')
        
        if data_path[-1] == '/':
            #remove the last /
            data_path = data_path[:-1]
            
        else:
            data_path = data_path
        

        self.data_path = data_path



    def make_path(self):



        os.makedirs(data_path+'/OPLS_DA_report', exist_ok=True)
        os.makedirs(data_path+'/OPLS_DA_report/main', exist_ok=True)
        os.makedirs(data_path+'/OPLS_DA_report/element', exist_ok=True)
        os.makedirs(data_path+'/OPLS_DA_report/element/hist_plot', exist_ok=True)
        os.makedirs(data_path+'/OPLS_DA_report/element/Lingress', exist_ok=True)
        os.makedirs(data_path+'/OPLS_DA_report/element/loading_plot', exist_ok=True)
        os.makedirs(data_path+'/OPLS_DA_report/element/s_plot', exist_ok=True)
        os.makedirs(data_path+'/OPLS_DA_report/element/score_plot', exist_ok=True)
        os.makedirs(data_path+'/OPLS_DA_report/element/VIP_score', exist_ok=True)


        #create dictionary to store the path
        path = {}
        path['main'] = data_path+'/OPLS_DA_report/main'
        path['element'] = data_path+'/OPLS_DA_report/element'
        path['hist_plot'] = data_path+'/OPLS_DA_report/element/hist_plot'
        path['Lingress'] = data_path+'/OPLS_DA_report/element/Lingress'
        path['loading_plot'] = data_path+'/OPLS_DA_report/element/loading_plot'
        path['s_plot'] = data_path+'/OPLS_DA_report/element/s_plot'
        path['score_plot'] = data_path+'/OPLS_DA_report/element/score_plot'
        path['VIP_score'] = data_path+'/OPLS_DA_report/element/VIP_score'

        self.path = path

    def get_path(self):
        path = self.path
        return path



class Normality_distribution:

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import scipy.stats as stats
    import pandas as pd

    def __init__(self, data: pd.DataFrame):
        self.data = data

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import scipy.stats as stats
        import pandas as pd

        """
        This function takes in a dataframe and a feature and returns the histogram and Q-Q plot of the feature.
        Parameters
        ----------
        data: pandas dataframe
            The dataframe to be used.
        feature: str
            The feature to be used.
        Normality_distribution(data, feature).plot_distribution()

        """
        n_features = data.shape[1]
        n_rows = data.shape[0]
        # check memory size for data
        def memory_size(X: pd.DataFrame) -> None:
            
            # unit of size
            size = ['B', 'KB', 'MB', 'GB', 'TB']
            X = X.memory_usage().sum()
            for i in range(len(size)):
                if X < 1024:
                    return f'{X:.2f} {size[i]}'
                X /= 1024
            return X
        sizes = memory_size(data)

        return print(f"Data has {n_features} features and {n_rows} samples. \n The memory size is {sizes}")

    def plot_distribution(self, feature):

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import scipy.stats as stats
        import pandas as pd
        
        data = self.data

        """
        This function takes in a dataframe and a feature and returns the histogram and Q-Q plot of the feature.
        Parameters
        ----------
        data: pandas dataframe
            The dataframe to be used.
        feature: str

        Normality_distribution(data).plot_distribution(feature)
        """


        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data[feature], kde=True)
        plt.title(f'Histogram of {feature}')

        plt.subplot(1, 2, 2)
        stats.probplot(data[feature], dist="norm", plot=plt)
        plt.title(f'Q-Q plot of {feature}')
        plt.show()

        return plt

    def pca_distributions(self):
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import scipy.stats as stats
        import pandas as pd
        """
        This function takes in a dataframe and a list of features and returns the histogram and Q-Q plot of the features.
        Parameters
        ----------
        data: pandas dataframe
            The dataframe to be used.
        features: list
            The list of features to be used.
        Normality_distribution.pca_distributions(data, features)
        """
        data = self.data

        from metbit import pca

        pca = pca(data , label = ["data" for x in range(data.shape[0])])
        pca.fit()
        scores = pca.get_scores()
        for feature in scores.columns[:2]:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(scores[feature], kde=True)
            plt.title(f'Histogram of {feature}')

            plt.subplot(1, 2, 2)
            stats.probplot(scores[feature], dist="norm", plot=plt)
            plt.title(f'Q-Q plot of {feature}')
            plt.show()

        return plt
      

class Normalise:

    import pandas as pd
    import numpy as np

    def __init__(self, data: pd.DataFrame, compute_missing: bool = True):
        """
        This function takes in a dataframe and returns the normalised dataframe.
        Parameters
        ----------
        data: pandas dataframe
            The dataframe to be used.
        Normalise(data).normalise()
        
        """
        
        if compute_missing:
            # Predict missing values using KNN
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=2)
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            self.data = data
        else:
            self.data = data
        
        n_features = data.shape[1]
        n_rows = data.shape[0]
        # check memory size for data
        def memory_size(X: pd.DataFrame) -> None:
            
            # unit of size
            size = ['B', 'KB', 'MB', 'GB', 'TB']
            X = X.memory_usage().sum()
            for i in range(len(size)):
                if X < 1024:
                    return f'{X:.2f} {size[i]}'
                X /= 1024
            return X
        sizes = memory_size(data)

        return print(f"Data has {n_features} features and {n_rows} samples. \n The memory size is {sizes}")


    def pqn_normalise(self, plot: bool = True):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        """
        This function returns the normalised dataframe using the PQN method.
        Parameters
        ----------
        data: pandas dataframe
            The dataframe to be used.
        plot: bool, optional
            Whether to plot the histograms of normalization factors and fold changes.
        """
        data = self.data
        features = data.columns

        median_spectra = data.median(axis=0)

        foldChangeMatrix = data.div(median_spectra, axis=1)
        # PQN normalisation with median
        pqn_coef = foldChangeMatrix.median(axis=1)

        norm_df = data.div(pqn_coef, axis=0)

        norm_df.columns = features
        norm_df.index = data.index   

        if plot:
            plt.figure()
            plt.hist(1/pqn_coef, bins=25)
            plt.xlabel("1/PQN Coefficient")
            plt.ylabel('Frequency')
            plt.title("Distribution of Normalisation factors")
            plt.show()

            # Truncate extreme values to narrow histogram range
            sample_to_plot = np.random.randint(0, data.shape[0])
            idx_to_plot = ((foldChangeMatrix.iloc[sample_to_plot, :] <= 5) & (foldChangeMatrix.iloc[sample_to_plot, :] >= -5 ))

            plt.figure()
            plt.title(f'Fold change to reference for sample: {sample_to_plot}')
            plt.xlabel("Fold Change to median")
            plt.ylabel("Frequency")
            plt.hist(foldChangeMatrix.loc[sample_to_plot, idx_to_plot], bins=100)
            plt.show()
        
        return norm_df

    def decimal_place_normalisation(self, decimals: int = 2):
        """
        This function returns the dataframe with values rounded to a specified number of decimal places.
        Parameters
        ----------
        decimals: int
            The number of decimal places to round to.
        """
        data = self.data.round(decimals)
        return data


    def z_score_normalisation(self):
        """
        This function returns the dataframe normalized using Z-Score.
        """
        from scipy.stats import zscore
        data = self.data.apply(zscore)

        return data

    def linear_normalisation(self):
        """
        This function returns the dataframe normalized using Min-Max (linear normalization).
        """
        data = self.data
        data = (data - data.min()) / (data.max() - data.min())
        
        return data

    def normalize_to_100(self):
        """
        This function returns the dataframe with values normalized to 100.
        """
        data = self.data
        data = (data / data.sum()) * 100
        
        return data

    def clipping_normalisation(self, lower: float, upper: float):
        """
        This function returns the dataframe with values clipped to the specified range.
        Parameters
        ----------
        lower: float
            The lower bound for clipping.
        upper: float
            The upper bound for clipping.
        """
        data = self.data.clip(lower, upper)
        
        return data

    def standard_deviation_normalisation(self):
        """
        This function returns the dataframe normalized using Standard Deviation.
        """
        data = self.data
        mean = data.mean()
        std = data.std()
        data = (data - mean) / std
        
        return data


class project_name_generator:

    def __init__():
        #random project name
        #get random time
        import random
        import time
        # Get local time
        current_time = time.localtime()
        # Set format for time
        time_format = time.strftime('%Y-%m-%d %H:%M:%S', current_time)
        project_names = [
                        "QuantumQuest",
                        "NebulaNet",
                        "StellarSync",
                        "AeroPulse",
                        "CyberCircuit",
                        "TerraTrack",
                        "HoloHive",
                        "PyroPixel",
                        "LunarLoom",
                        "ZenithZero",
                        "BlazeBeacon",
                        "AquaArise",
                        "EchoEclipse",
                        "FusionForge",
                        "OrbitOpus",
                        "PrismPortal",
                        "NimbusNexus",
                        "AstroArc",
                        "VoltVoyage",
                        "OmniOrbit",
                        "PulsePioneer",
                        "VortexVoyage",
                        "GalacticGrid",
                        "SolarSpectrum",
                        "Satternlite",
                        "StarSpectrum",
                        "SpaceSpectrum",
                        "GalacticSpectrum"
                        ]

        project_name = time_format + '_' + random.choice(project_names)
        return project_name


class lazy_opls_da:

    
    import os
    from glob import glob
    import pandas as pd
    import numpy as np
    import random
    from metbit import opls_da

    from metbit import project_name_generator
    

    def __init__(self, data: pd.DataFrame, groups: list, working_dir: str, n_components: int = 2, scaling: str = 'pareto', 
                    estimator: str = 'opls', kfold: int = 3, random_state: int = 94, auto_ncomp: bool = True,  
                    permutation: bool = True, 
                    VIP: bool = True, 
                    linear_regression: bool = True) -> None:

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

        self.random_state = random_state        
        self.estimator = estimator
        self.scale = scaling
        self.kfold = kfold
        self.auto_ncomp = auto_ncomp
        

        data['Class'] = groups
        self.data = data

        self.permutation = permutation
        if permutation == True:
            self.n_permutataion = int(input('Enter the number of permutation: '))
            self.n_jobs = int(input('Enter the number of jobs: '))
        else:
            pass
        
        self.VIP = VIP
        if VIP == True:
            self.VIP_threshold = float(input('Enter the VIP threshold: '))

        self.linear_regression = linear_regression
        if linear_regression == True:
            self.FC_threshold = float(input('Enter the fold change threshold: '))
            self.p_val_threshold = float(input('Enter the p-value threshold: '))

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
            folder_name_data = []
            if permutation == True:
                folder_name_plot.append('hist_plot')
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

        #Create dictionary to store the path
        path = {}
        for i in dir:
            path[i.split('/')[-2]] = i

        self.color_map = color_map
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
        Permutation: {permutation}
        VIP: {VIP}
        Linear regression: {linear_regression}
        """

        return print(Summary)



    def fit(self, marker_color: dict = None) -> None:

        from metbit import opls_da
        from lingress import lin_regression
        from metbit import lazypair

     
        data = self.data
        n_components = self.n_components
        path = self.path
        color_map = self.color_map
        scale = self.scale

        marker_color = marker_color


        #Create object attribute
        lazy = lazypair(data, 'Class')
        data_list = lazy.get_dataset()
        name_save = lazy.get_name()

        for i in range(len(data_list)):
            
            df = data_list[i]
            name = name_save[i]

            X = df.drop('Class', axis=1)
            y = df['Class']
            feature_names = X.columns
            # Check if feature names can be converted to float
            try:
                feature_names = feature_names(float).tolist()
            except:
                feature_names = feature_names.tolist()

            #OPLS-DA
            oplsda_mod = opls_da(X=X, y=y, features_name = feature_names, n_components=n_components, scale=scale, 
                                    estimator=self.estimator, kfold=self.kfold, random_state=self.random_state, 
                                    auto_ncomp = self.auto_ncomp)
            oplsda_mod.fit()

            #Score plot
            oplsda_mod.plot_oplsda_scores(color_dict=marker_color).write_html(path['score_plot'] + name + '_score_plot.html')
            oplsda_mod.plot_oplsda_scores(color_dict=marker_color).write_image(path['score_plot'] + name + '_score_plot.png')

            #Loading plot
            oplsda_mod.plot_loading().write_html(path['loading_plot'] + name + '_loading_plot.html')
            oplsda_mod.plot_loading().write_image(path['loading_plot'] + name + '_loading_plot.png')

            #S plot
            oplsda_mod.plot_s_scores().write_html(path['s_plot'] + name + '_s_plot.html')
            oplsda_mod.plot_s_scores().write_image(path['s_plot'] + name + '_s_plot.png')

            #VIP score plot
            if self.VIP == True:
                oplsda_mod.vip_scores()
                oplsda_mod.get_vip_scores().to_csv(path['VIP_scores'] + name + '_VIP_scores.csv')
                oplsda_mod.vip_plot(threshold=self.VIP_threshold).write_html(path['VIP_score_plot'] + name + '_VIP_score_plot.html')
                oplsda_mod.vip_plot(threshold=self.VIP_threshold).write_image(path['VIP_score_plot'] + name + '_VIP_score_plot.png')
            else:
                pass

            #Permutation test
            if self.permutation == True:
                oplsda_mod.permutation_test(n_permutations=self.n_permutataion, n_jobs=self.n_jobs)
                oplsda_mod.plot_hist().write_html(path['hist_plot'] + name + '_hist_plot.html')
                oplsda_mod.plot_hist().write_image(path['hist_plot'] + name + '_hist_plot.png')
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

