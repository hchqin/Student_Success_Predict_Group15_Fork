import pandas as pd
import pytest
import sys
import os
import numpy as np

# Import the get_high_correlations function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.high_corr_extract import get_high_correlations  

# Test data
df_numeric = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [4, 3, 2, 1],
    'C': [1, 0.5, 3, 4]
})

df_empty = pd.DataFrame()

df_mixed_types = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['a', 'b', 'c', 'd']
})

df_single_column = pd.DataFrame({
    'A': [1, 2, 3, 4]
})

df_known_corr = pd.DataFrame({
    'X1': [1, 2, 3, 4, 5],
    'X2': [2, 4.5, 6, 8, 10],  
    'Y1': [5, 4, 3, 2, 1],
    'Y2': [1, 2, 1, 2, 1]   
})

df_with_nan = pd.DataFrame({
    'A': [1, np.nan, 3, 4],
    'B': [4, 3, 2, 1],
    'C': [np.nan, 0.5, 3, 4]
})

# Test cases
def test_high_correlation_with_numeric_data():
    result = get_high_correlations(df_numeric)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

def test_high_correlation_with_empty_dataframe():
    with pytest.raises(ValueError):
        get_high_correlations(df_empty)

def test_high_correlation_with_mixed_types():
    with pytest.raises(ValueError):
        get_high_correlations(df_mixed_types)

def test_high_correlation_with_single_column():
    with pytest.raises(ValueError):
        get_high_correlations(df_single_column)

def test_correct_pair_identification():
    result = get_high_correlations(df_known_corr, threshold=0.9)
    assert ('X1', 'X2') in [(row['Variable 1'], row['Variable 2']) for _, row in result.iterrows()] or ('X2', 'X1') in [(row['Variable 1'], row['Variable 2']) for _, row in result.iterrows()]

def test_threshold_out_of_range():
    with pytest.raises(ValueError):
        get_high_correlations(df_numeric, threshold=1.1)
    with pytest.raises(ValueError):
        get_high_correlations(df_numeric, threshold=-1.1)

def test_negative_correlation():
    df_negative_corr = pd.DataFrame({
        'X': [1, 2, 3, 4, 5],
        'Y': [-1, -2, -3, -4, -5]
    })
    result = get_high_correlations(df_negative_corr, threshold=0.9)
    assert ('X', 'Y') in [(row['Variable 1'], row['Variable 2']) for _, row in result.iterrows()] or ('Y', 'X') in [(row['Variable 1'], row['Variable 2']) for _, row in result.iterrows()]

def test_dataframe_with_nan():
    with pytest.raises(ValueError):
        get_high_correlations(df_with_nan)

def test_threshold_edge_cases():
    result = get_high_correlations(df_numeric, threshold=1)
    assert isinstance(result, pd.DataFrame)

    result = get_high_correlations(df_numeric, threshold=-1)
    assert isinstance(result, pd.DataFrame)
