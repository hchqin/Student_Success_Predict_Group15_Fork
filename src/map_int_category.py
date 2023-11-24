import pandas as pd


def map_int_categories(dataframe, column_name, text_mapping):
    """
    Maps integer identifiers to descriptive names in a pandas DataFrame.

    This function replaces integer identifiers in a specified column of the DataFrame
    with corresponding descriptive text based on a provided mapping. It performs 
    several checks to ensure the validity of the input arguments and the data in the 
    specified column.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the integer identifiers to be mapped.
    column_name : str
        The name of the column in the DataFrame where the mapping is to be applied.
    text_mapping : dict
        A dictionary mapping integer identifiers to their corresponding text descriptions.
        Keys must be integers, and values must be strings.

    Returns:
    -------
    pandas.DataFrame
        The DataFrame with integer identifiers in the specified column replaced by 
        their corresponding text descriptions.

    Raises:
    ------
    TypeError: If the input 'dataframe' is not a pandas DataFrame, if 'column_name' 
               is not a string, or if 'text_mapping' is not a dictionary.
    ValueError: If the specified 'column_name' does not exist in the DataFrame,
                if keys or values in 'text_mapping' are not of the correct types 
                (keys must be integers, values must be strings), or if any value in 
                the specified column is not an integer.

    Examples:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'Category': [1, 2, 3]})
    >>> text_mapping = {1: 'Category A', 2: 'Category B', 3: 'Category C'}
    >>> updated_df = map_int_categories(df, 'Category', text_mapping)
    >>> print(updated_df)

    Notes:
    -----
    This function modifies the input DataFrame in-place. Ensure that the provided 
    mapping covers all unique integer values present in the specified column.
    """

    
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("The 'dataframe' argument must be a pandas DataFrame.")

    
    if not isinstance(column_name, str):
        raise TypeError("The 'column_name' argument must be a string.")

    
    if not isinstance(text_mapping, dict):
        raise TypeError("The 'text_mapping' argument must be a dictionary.")

    
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    
    if not all(isinstance(key, int) and isinstance(value, str) for key, value in text_mapping.items()):
        raise ValueError("All keys in 'text_mapping' must be integers and all values must be strings.")

    
    if not all(isinstance(x, int) for x in dataframe[column_name]):
        raise ValueError(f"All values in column '{column_name}' must be integers.")

    dataframe[column_name] = dataframe[column_name].map(text_mapping)

    return dataframe