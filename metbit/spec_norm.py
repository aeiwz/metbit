
class Normalization:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np


    def __init__(self):
        pass

    def pqn_normalization(df, plot=False):
        """
        Probabilistic Quotient Normalization (PQN) method
        """

        import numpy as np
        import pandas as pd
        #check data type of input data id dataframe or not
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        try:
            nom_arr = df.values
            feature_ = df.columns

            # Calculate the median of each row
            median = np.median(nom_arr, axis=0, keepdims=True)

            # Divide each row by its median
            df_norm = nom_arr / median
            df_norm = pd.DataFrame(df_norm, columns=feature_)

            if plot == True:
                #histogram sub-plot of PQN normalized data with fold change of original data and PQN normalized data
                #Calculate fold change of original data and PQN normalized data
                fold_change = nom_arr / df_norm.values
                


                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                fig.suptitle('PQN Normalization')
                axs[0].hist(df_norm.values.flatten(), bins=50)
                axs[0].set_title('Histogram of PQN normalized data')
                axs[0].set_xlabel('Intensity')
                axs[0].set_ylabel('Frequency')
                axs[1].hist(fold_change.flatten(), bins=50)
                axs[1].set_title('Histogram of fold change data')
                axs[1].set_xlabel('Intensity')
                axs[1].set_ylabel('Frequency')
                axs[1].set_xlim(0, 10)
                axs[1].set_ylim(0, 20000)
                plt.show()


        except:
            print("Error: Please check your input data")

        return df_norm
    
    def snv_normalization(df):
        """
        Standard Normal Variate (SNV) method
        """
        import numpy as np
        import pandas as pd
        #check data type of input data id dataframe or not
        
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        try:
            nom_arr = df.values
            feature_ = df.columns

            # Calculate the mean of each row
            mean = np.mean(nom_arr, axis=0, keepdims=True)

            # Calculate the standard deviation of each row
            std = np.std(nom_arr, axis=0, keepdims=True)

            # Subtract the mean from each row
            df_norm = nom_arr - mean

            # Divide each row by its standard deviation
            df_norm = df_norm / std
            df_norm = pd.DataFrame(df_norm, columns=feature_)
        except:
            print("Error: Please check your input data")

        return df_norm

    def msc_normalization(df):
        """
        Multiplicative Scatter Correction (MSC) method
        """
        import numpy as np
        import pandas as pd
        #check data type of input data id dataframe or not
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        try:
            nom_arr = df.values
            feature_ = df.columns

            # Calculate the mean of each row
            mean = np.mean(nom_arr, axis=0, keepdims=True)

            # Subtract the mean from each row
            df_norm = nom_arr - mean

            # Calculate the standard deviation of each row
            std = np.std(df_norm, axis=1, keepdims=True)

            # Divide each row by its standard deviation
            df_norm = df_norm / std

            # Calculate the mean of each column
            mean = np.mean(df_norm, axis=0, keepdims=True)

            # Subtract the mean from each column
            df_norm = df_norm - mean
            df_norm = pd.DataFrame(df_norm, columns=feature_)
        except:
            print("Error: Please check your input data")

        return df_norm

    def snv_msc_normalization(df):
        """
        SNV-MSC method
        """
        import numpy as np
        import pandas as pd
        #check data type of input data id dataframe or not
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        try:
            nom_arr = df.values
            feature_ = df.columns

            # Calculate the mean of each row
            mean = np.mean(nom_arr, axis=0, keepdims=True)

            # Calculate the standard deviation of each row
            std = np.std(nom_arr, axis=0, keepdims=True)

            # Subtract the mean from each row
            df_norm = nom_arr - mean

            # Divide each row by its standard deviation
            df_norm = df_norm / std

            # Calculate the mean of each column
            mean = np.mean(df_norm, axis=0, keepdims=True)

            # Subtract the mean from each column
            df_norm = df_norm - mean
            df_norm = pd.DataFrame(df_norm, columns=feature_)
        except:
            print("Error: Please check your input data")

        return df_norm

    def snv_pqn_normalization(df):
        """
        SNV-PQN method
        """
        import numpy as np
        import pandas as pd
        #check data type of input data id dataframe or not
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        try:
            nom_arr = df.values
            feature_ = df.columns

            # Calculate the median of each row
            median = np.median(nom_arr, axis=1, keepdims=True)

            # Divide each row by its median
            df_norm = nom_arr / median

            # Calculate the mean of each row
            mean = np.mean(df_norm, axis=1, keepdims=True)

            # Subtract the mean from each row
            df_norm = df_norm - mean
            df_norm = pd.DataFrame(df_norm, columns=feature_)
        except:
            print("Error: Please check your input data")

        return df_norm

    def snv_msc_pqn_normalization(df):
        """
        SNV-MSC-PQN method
        """
        import numpy as np
        import pandas as pd
        #check data type of input data id dataframe or not
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        try:
            nom_arr = df.values
            feature_ = df.columns

            # Calculate the median of each row
            median = np.median(nom_arr, axis=1, keepdims=True)

            # Divide each row by its median
            df_norm = nom_arr / median

            # Calculate the mean of each row
            mean = np.mean(df_norm, axis=1, keepdims=True)

            # Subtract the mean from each row
            df_norm = df_norm - mean

            # Calculate the mean of each column
            mean = np.mean(df_norm, axis=0, keepdims=True)

            # Subtract the mean from each column
            df_norm = df_norm - mean
            df_norm = pd.DataFrame(df_norm, columns=feature_)
        except:
            print("Error: Please check your input data")

        return df_norm

   