import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()

def plot_numeric_feature_distribution(dataframe, numeric_features, target_column):
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


        
