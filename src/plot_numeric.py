import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()

def plot_numeric_feature_distribution(dataframe, numeric_features, target_column):

    """
    Create a faceted bar chart to visualize the distribution of numeric features.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data to analyze.
    numeric_features : list
        List of column names representing numeric features to visualize.
    target_column : str
        The name of the column in the DataFrame containing target labels.

    Returns:
    -------
    alt.Chart
        An Altair Chart object representing the faceted bar chart.

    Raises:
    ------
    ValueError
        If 'target_column' is not present in the DataFrame.
    TypeError
        If 'numeric_features' is not a list or if any of the features are not present in the DataFrame.

    Examples:
    --------
    >>> import pandas as pd
    >>> import altair as alt
    >>> data = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your dataset file
    >>> chart = plot_numeric_feature_distribution(data, ['feature1', 'feature2'], 'target_column')
    >>> chart.show()

    Notes:
    -----
    This function uses Altair to create a faceted bar chart, visualizing the distribution
    of numeric features grouped by the target column.

    """
    # Parameter checks
    if target_column not in dataframe.columns:
        raise ValueError(f"The target column '{target_column}' is not present in the DataFrame.")

    if not isinstance(numeric_features, list):
        raise TypeError("'numeric_features' must be a list.")


    # Melt the DataFrame to long format for Altair's facet
    melted_df = pd.melt(dataframe, id_vars=[target_column], value_vars=numeric_features)

    chart = alt.Chart(melted_df).mark_bar().encode(
        x = alt.X('value:Q', bin=alt.Bin(maxbins=50), title='Value', axis=alt.Axis(labels=True)),
        y = alt.Y('count()', title='Count'),
        color = alt.Color(target_column + ':N', title=target_column),
        facet = alt.Facet('variable:N', columns=2, title='Numeric Feature')
    ).properties(
        width=350,
        height=150
    ).resolve_scale(x='independent')

    return chart


        
