
class Normalization:
    """
    A class for performing various normalization methods, including 
    Probabilistic Quotient Normalization (PQN), Standard Normal Variate 
    (SNV), Multiplicative Scatter Correction (MSC), and their combinations.

    Methods:
    --------
    pqn_normalization(df):
        Applies Probabilistic Quotient Normalization (PQN) to the input dataframe.
    
    snv_normalization(df):
        Applies Standard Normal Variate (SNV) normalization to the input dataframe.

    msc_normalization(df):
        Applies Multiplicative Scatter Correction (MSC) normalization to the input dataframe.

    snv_msc_normalization(df):
        Applies SNV followed by MSC normalization to the input dataframe.

    snv_pqn_normalization(df):
        Applies SNV followed by PQN normalization to the input dataframe.

    snv_msc_pqn_normalization(df):
        Applies SNV followed by MSC and then PQN normalization to the input dataframe.
    """


    def __init__(self):
        """Initializes the Normalization class."""
        pass

    @staticmethod
    def pqn_normalization(df):
        """
        Perform Probabilistic Quotient Normalization (PQN) on a dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame or numpy.ndarray
            The input data to normalize. Each column represents a feature.

        Returns:
        --------
        df_norm : pandas.DataFrame
            The PQN normalized dataframe.

        Raises:
        -------
        TypeError: 
            If input is not a pandas DataFrame or cannot be converted to one.
        """
        import numpy as np
        import pandas as pd

        # Check if the input is a DataFrame, if not, attempt to convert it
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise TypeError(f"Input data is not a valid DataFrame or cannot be converted: {e}")

        try:
            # Get the numpy array and column names
            X_ = df.values
            feature_ = df.columns

            # Calculate the median across the rows (per column)
            median_spectra = np.median(X_, axis=0, keepdims=True)

            # Calculate the fold change matrix
            foldChangeMatrix = X_ / median_spectra

            # Calculate the PQN normalization coefficients (median fold change per row)
            pqn_coef = np.nanmedian(foldChangeMatrix, axis=1)

            # Normalize the data by dividing each row by its PQN coefficient
            # Reshape pqn_coef to a column vector (if needed) to match the shape of X_
            norm_X = X_ / pqn_coef[:, np.newaxis]

            # Convert the normalized data back to a DataFrame
            df_norm = pd.DataFrame(norm_X, columns=feature_)

        except Exception as e:
            raise ValueError(f"An error occurred during normalization: {e}")

        return df_norm
    
    def snv_normalization(df):
        """
        Apply Standard Normal Variate (SNV) normalization to a dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame or numpy.ndarray
            The input data to normalize. Each column represents a feature.

        Returns:
        --------
        df_norm : pandas.DataFrame
            The SNV normalized dataframe.

        Raises:
        -------
        TypeError:
            If input is not a pandas DataFrame or cannot be converted to one.
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
        Apply Multiplicative Scatter Correction (MSC) normalization to a dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame or numpy.ndarray
            The input data to normalize. Each column represents a feature.

        Returns:
        --------
        df_norm : pandas.DataFrame
            The MSC normalized dataframe.

        Raises:
        -------
        TypeError:
            If input is not a pandas DataFrame or cannot be converted to one.
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
        Apply SNV followed by MSC normalization to a dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame or numpy.ndarray
            The input data to normalize. Each column represents a feature.

        Returns:
        --------
        df_norm : pandas.DataFrame
            The SNV-MSC normalized dataframe.

        Raises:
        -------
        TypeError:
            If input is not a pandas DataFrame or cannot be converted to one.
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
        Apply SNV followed by PQN normalization to a dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame or numpy.ndarray
            The input data to normalize. Each column represents a feature.

        Returns:
        --------
        df_norm : pandas.DataFrame
            The SNV-PQN normalized dataframe.

        Raises:
        -------
        TypeError:
            If input is not a pandas DataFrame or cannot be converted to one.
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
        Apply SNV followed by MSC and then PQN normalization to a dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame or numpy.ndarray
            The input data to normalize. Each column represents a feature.

        Returns:
        --------
        df_norm : pandas.DataFrame
            The SNV-MSC-PQN normalized dataframe.

        Raises:
        -------
        TypeError:
            If input is not a pandas DataFrame or cannot be converted to one.
        ValueError:
            If an error occurs during the normalization process.
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

   