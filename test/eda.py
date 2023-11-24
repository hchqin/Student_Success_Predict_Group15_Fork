import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_analysis_EDA import calculate_correlation_df

# Create a sample DataFrame for testing
data = {
    'A': [1, 2, 3, 4],
    'B': [2, 4, 6, 8],
    'C': [5, 5, 5, 5]
    }
df = pd.DataFrame(data)

def test_correlation_calculation_with_threshold_shape():
    threshold = 0.7
    result_df = calculate_correlation_df(df, threshold)

    # Check if the result is a DataFrame
    assert isinstance(result_df, pd.DataFrame), "it should return a dataframe"
    # Check if the expected columns are present
    assert 'Variable 1' in result_df.columns, "the first variable is missing"
    assert 'Variable 2' in result_df.columns, "the second variable is missing"
    assert 'Correlation' in result_df.columns, "the correlation is missing"

def test_correlation_calculation_accuracy():
    result_df = calculate_correlation_df(df)
    assert isinstance(result_df, pd.DataFrame), "The function should return a dataframe"
    assert not result_df.empty, "The dataframe should not be empty"
    assert 'Variable 1' in result_df.columns, "The first variable is missing"
    assert 'Variable 2' in result_df.columns, "The second variable is missing"
    assert 'Correlation' in result_df.columns, "The correlation is missing"
    assert result_df['Correlation'].abs().iloc[0] == 1.0, "The correlation calculated is wrong"

def test_correlation_calculation_without_threshold_shape():
    result_df = calculate_correlation_df(df)

    assert isinstance(result_df, pd.DataFrame), "it should return a dataframe"
    assert 'Variable 1' in result_df.columns, "the first variable is missing"
    assert 'Variable 2' in result_df.columns, "the second variable is missing"
    assert 'Correlation' in result_df.columns, "the correlation is missing"
