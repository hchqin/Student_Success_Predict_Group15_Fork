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
