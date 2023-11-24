import pandas as pd
import numpy as np

def get_high_correlations(df, threshold=0.5):
    """
    Computes the correlation matrix for numeric features in the given DataFrame and
    returns pairs of features where the absolute value of their correlation 
    coefficient exceeds a specified threshold. The pairs containing the same feature 
    twice are not included, even if their correlation coefficient is 1.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing numeric features.
    threshold : float, optional (default=0.5)
        The threshold for selecting high correlations. Only pairs with an absolute 
        correlation coefficient greater than or equal to this value are returned.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing pairs of variables with high correlation. 
        Columns are: ['Variable 1', 'Variable 2', 'Correlation'].

    Raises:
    ------
    ValueError: If the input is not a DataFrame, the DataFrame is empty,
                if non-numeric columns are present in the DataFrame, or
                if the threshold is not within the range -1 to 1.
                the DataFrame contains NaN values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Check if threshold is within the valid range
    if not (-1 <= threshold <= 1):
        raise ValueError("Threshold must be between -1 and 1.")

    # Check if DataFrame is non-empty
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Check for NaN values in the DataFrame
    if df.isna().any().any():
        raise ValueError("DataFrame contains NaN values.")

    # Check for non-numeric columns and exclude them
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] != df.shape[1]:
        raise ValueError("Non-numeric columns found in the DataFrame.")

    # Check if DataFrame has at least two numeric columns
    if numeric_df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two numeric columns for correlation calculation.")

    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()
