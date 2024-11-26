# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

__auther__ ='aeiwz'
author_email='theerayut_aeiw_123@hotmail.com'

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
        if meta[column_name].nunique() < 2:
            raise ValueError("Group should contain at least 2 groups")
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


def project_name_generator():
    #random project name
    #get random time
    import random
    import time
    # Get local time
    current_time = time.localtime()
    # Set format for time
    time_format = time.strftime('%Y-%m-%d_%H-%M-%S', current_time)
    project_names = [
    "ApolloPulse", "OrbitOmni", "NebulaNexus", "StarStream", "CometCore",
    "AstralAxis", "CelestialSync", "MeteorMerge", "GalaxusGate", "StellarScope",
    "NovaNest", "SpectraSphere", "IonIgnite", "QuasarQuest", "CosmosCircuit",
    "OrbitOxide", "CelestialCircuit", "GalaxyGrid", "ApolloAlign", "StellarSignal",
    "HyperHalo", "LunarLattice", "StarForge", "NebulaNode", "AstrumAxis",
    "OrbitOps", "GalacticGate", "MeteorMap", "CosmicCore", "SolsticeSync",
    "EclipseEcho", "CelestiaConnect", "ZenithZone", "VoidVector", "AstroAlign",
    "PlasmaPath", "OrbitOscillator", "CometCatalyst", "AetherArc", "VoidVelocity",
    "PulsarPulse", "StellarSail", "AstralAnchor", "PhotonPath", "VortexVector",
    "OrbitOptic", "NovaNetwork", "StarSphere", "EchoEnergy", "ChronoCelestial",
    "QuantumVoyage", "NebulaNexus", "StellarSync", "AstroArray", "GalacticGlow",
    "PhotonPulse", "QuantumQuasar", "CelestialCircuit", "NovaNucleus", "CosmicCascade",
    "StellarSpire", "AstroArc", "NebulaNode", "QuasarQuest", "PlasmaPioneer",
    "InfinityIon", "OrbitOracle", "CelestialClimb", "QuantumQuest", "StarlightSync",
    "GalaxiaGlimmer", "PulsarPath", "CosmosCircuit", "QuantumSphere", "AstroAxis",
    "HyperHelix", "StellarScope", "CelestiaChrono", "EclipseEngine", "QuantaCove",
    "OrbitOrigin", "MeteorMind", "PhotonPath", "StarSystem", "ChronoCelestial",
    "VoidVector", "GalaxyGate", "CosmicCircuit", "AetherArc", "LunarLoom",
    "QuantaCluster", "NovaNest", "SpectraSphere", "NebulaNavigator", "PulsarPeak",
    "OrbitOdyssey", "CosmicConduit", "TerraTrajectory", "StellarStrata", "VoidVoyager",
    "EclipseEcho", "ZenithZone", "CelestialConnect", "AstroAlign", "IonIgnite",
    "AetherAtlas", "GalaxusGrid", "QuantaQuay", "HorizonHalo", "AstralApex",
    "ZenithZephyr", "GalacticGlide", "CelestialSync", "PlasmaPulse", "QuantumPulse",
    "NebulaNebula", "AstroAlign", "CometClimb", "GalacticGaze", "LunarLink",
    "StellarSplice", "EclipseEngine", "NovaNode", "PulsarPilot", "PhotonPortal",
    "QuantaQuest", "CelestialClimb", "GalacticGlider", "AstralAnchor", "ZenithZero",
    "VortexVector", "PulsarPathfinder", "IonInfinity", "ChronoCircuit", "QuantumQuay",
    "NebulaNucleus", "StarSphere", "GalacticGate", "InfinityIris", "HorizonHub",
    "StellarSignal", "NovaNexus", "CosmosCore", "GalaxiaGrid", "CelestialCompass",
    "PulsarPioneer", "AstralAether", "PlasmaPeak", "OrbitOpus", "AetherArcadia",
    "CelestialCircuitry", "PhotonPeak", "ZenithZone", "VoidVoyager", "QuasarCove"
    ]

    project_name = time_format + '_' + random.choice(project_names)
    return project_name



import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations
import numpy as np
def boxplot_stats(
    df, 
    x_col, 
    y_col, 
    group_order=None, 
    custom_colors=None, 
    p_value_threshold=0.05, 
    annotate_style="value", 
    figure_size=(800, 600),
    y_offset_factor=0.05,  # Use a factor to scale offset based on y-axis range
    show_non_significant=True,
    title_=None,
    y_label=None,
    x_label=None,
    fig_height=800,
    fig_width=600
):
    """
    Creates a box plot with tiered p-value annotations with improved y-axis positioning.

    Parameters:
    - df (DataFrame): The input data.
    - x_col (str): Column name for categorical grouping.
    - y_col (str): Column name for numerical values.
    - group_order (list, optional): Order of groups for visualization.
    - custom_colors (dict, optional): Custom colors for each group.
    - p_value_threshold (float, optional): Threshold for significance annotations.
    - annotate_style (str, optional): 'value' for p-value text, 'symbol' for stars ('*').
    - figure_size (tuple, optional): Width and height of the figure.
    - y_offset_factor (float, optional): Scaling factor for vertical spacing based on y-axis range.



    # Example usage
    data = {
        "treatment": ["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10,
        "score": [0.5, 0.7, 0.6, 0.5, 0.6, 0.8, 0.5, 0.6, 0.7, 0.8, 0.5, 0.7, 0.6, 0.5, 0.6, 0.8, 0.5, 0.6, 0.7, 0.8,
                0.40, 0.45, 0.50, 0.48, 0.41, 0.42, 0.47, 0.44, 0.45, 0.43, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4],
    }

    df = pd.DataFrame(data)
    custom_colors = {"A": "red", "B": "blue", "C": "green", "D": "purple"}
    fig = boxplot_stats(
        df, 
        x_col="treatment", 
        y_col="score", 
        group_order=["A", "B", "C", "D"], 
        custom_colors=custom_colors, 
        p_value_threshold=0.05, 
        annotate_style="symbol", 
        fig_height=800,
        fig_width=600,
        show_non_significant=True
    )
    fig.show()

    Returns:
    - fig (Figure): The Plotly figure object.
    """
    # Basic setup
    if title_ is None:
        title_ = y_col

    y_label = y_label or y_col
    x_label = x_label or x_col

    # Calculate dynamic y-offset based on data range
    y_range = df[y_col].max() - df[y_col].min()
    y_offset = y_offset_factor * y_range

    # Group data and calculate p-values
    grouped = df.groupby(x_col)[y_col]
    if group_order is None:
        group_order = list(grouped.groups.keys())
    comparisons = list(combinations(group_order, 2))
    p_values = []
    for g1, g2 in comparisons:
        t_stat, p_val = ttest_ind(grouped.get_group(g1), grouped.get_group(g2))
        p_values.append((g1, g2, p_val))

    # Create box plot
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col, 
        color=x_col, 
        points="all", 
        category_orders={x_col: group_order}, 
        color_discrete_map=custom_colors
    )

    # Add p-value annotations with dynamic y-offset
    max_y = df[y_col].max()
    annotations = []
    lines = []

    for i, (g1, g2, p_val) in enumerate(p_values):
        y_pos = max_y + (i + 1) * y_offset  # Dynamically adjusted based on data
        if annotate_style == "value":
            p_text = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        elif annotate_style == "symbol":
            if p_val < p_value_threshold:
                p_text = "*" if p_val < 0.05 else ""
                p_text += "*" if p_val < 0.01 else ""
                p_text += "*" if p_val < 0.001 else ""
            else:
                if show_non_significant:
                    p_text = "ns"
                else:
                    continue
        else:
            raise ValueError("Invalid annotate_style. Use 'value' or 'symbol'.")

        # Add horizontal line for the comparison
        lines.append(
            go.Scatter(
                x=[g1, g1, g2, g2], 
                y=[y_pos, y_pos + y_offset * 0.5, y_pos + y_offset * 0.5, y_pos], 
                mode="lines",
                line=dict(color="black", width=1),
                hoverinfo="none",
            )
        )
        
        # Add text annotation for the p-value
        annotations.append(
            dict(
                xref="x",
                yref="y",
                x=(group_order.index(g1) + group_order.index(g2)) / 2,
                y=y_pos + y_offset * 0.75,  # Position text above line
                text=p_text,
                showarrow=False,
                font=dict(size=12),
            )
        )

    # Add lines to the figure
    for line in lines:
        fig.add_trace(line)

    # Update layout with annotations and figure size
    fig.update_layout(
        annotations=annotations,
        title=f'<b>{title_}</b>',
        yaxis_title=y_label,
        xaxis_title=x_label,
        legend_title=x_col,
        width=fig_width,
        height=fig_height,
        showlegend=False
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    #update title position to center
    fig.update_layout(
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

    

    return fig