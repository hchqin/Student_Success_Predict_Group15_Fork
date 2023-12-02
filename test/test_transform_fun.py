import pandas as pd
import numpy as np
import pytest
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Import the get_high_correlations function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.transform_data import transform_data  

# Test data
df_simple = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
})

# Simple Preprocessor
simple_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['A', 'B'])
    ],
    remainder='passthrough'
)

# Test cases
def test_transform_data_with_simple_data():
    result = transform_data(df_simple, simple_preprocessor)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.shape == df_simple.shape

def test_transform_data_with_invalid_input():
    with pytest.raises(ValueError):
        transform_data("not_a_dataframe", simple_preprocessor)

def test_feature_names_match():
    transformed_df = transform_data(df_simple, simple_preprocessor)

    expected_feature_names = simple_preprocessor.get_feature_names_out()

    assert list(transformed_df.columns) == list(expected_feature_names), "Column names do not match the expected feature names"