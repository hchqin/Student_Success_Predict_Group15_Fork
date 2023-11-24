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