import pandas as pd


def map_int_categories(dataframe, column_name, text_mapping):


    

    dataframe[column_name] = dataframe[column_name].map(text_mapping)

    return dataframe