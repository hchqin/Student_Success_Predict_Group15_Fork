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

    # Check if keys and values in text_mapping are of correct types
    if not all(isinstance(key, int) and isinstance(value, str) for key, value in text_mapping.items()):
        raise ValueError("All keys in 'text_mapping' must be integers and all values must be strings.")

    # Check for numeric or None values in the specified column
    if not all(isinstance(x, int) for x in dataframe[column_name]):
        raise ValueError(f"All values in column '{column_name}' must be integers.")

    dataframe[column_name] = dataframe[column_name].map(text_mapping)

    return dataframe