import pandas as pd
import altair as alt
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.plot_numeric import plot_numeric_feature_distribution

# Toy data
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 3, 4, 5, 6],
    'target': ['yes', 'no', 'yes', 'no', 'yes']
}
df = pd.DataFrame(data)

def test_altair_object():
    chart = plot_numeric_feature_distribution(df, ['feature1', 'feature2'], 'target')
    assert isinstance(chart, alt.Chart), "Output should be an Altair Chart object"

def test_melting_data():
    chart = plot_numeric_feature_distribution(df, ['feature1', 'feature2'], 'target')
    assert chart.data.equals(pd.melt(df, id_vars=['target'], value_vars=['feature1', 'feature2'])), "Data should be correctly transformed"
