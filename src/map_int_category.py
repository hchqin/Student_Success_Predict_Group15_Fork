import pandas as pd


def map_int_categories(dataframe, column_name, text_mapping):


    # Check if the input dataframe is a pandas DataFrame
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("The 'dataframe' argument must be a pandas DataFrame.")

    # Check if the column_name is a string
    if not isinstance(column_name, str):
        raise TypeError("The 'column_name' argument must be a string.")

    # Check if the text_mapping is a dictionary
    if not isinstance(text_mapping, dict):
        raise TypeError("The 'text_mapping' argument must be a dictionary.")

    # Check if the specified column exists
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    dataframe[column_name] = dataframe[column_name].map(text_mapping)

    return dataframe