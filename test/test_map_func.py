import pandas as pd
import pytest
import sys
import os

# Import the map_int_categories function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.category_mapping_func import map_int_categories

# Test data and mappings
course_mapping = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    # ... other mappings
}


df_with_valid_course_codes = pd.DataFrame({'Course': [33, 171]})
df_with_invalid_course_codes = pd.DataFrame({'Course': [99999]})
df_empty = pd.DataFrame({'Course': []})
mixed_course_codes_df = pd.DataFrame({'Course': [33, 99999, 171]})
non_numeric_course_df = pd.DataFrame({'Course': ['A', 'B', 'C']})
df_with_null_values = pd.DataFrame({'Course': [33, None, 171]})

# Expected outputs
df_with_valid_course_codes_mapped = pd.DataFrame({'Course': ['Biofuel Production Technologies', 'Animation and Multimedia Design']})
df_with_invalid_course_codes_mapped = pd.DataFrame({'Course': [None]})
df_empty_mapped = pd.DataFrame({'Course': pd.Series([], dtype='object')})
mixed_course_codes_mapped = pd.DataFrame({'Course': ['Biofuel Production Technologies', None, 'Animation and Multimedia Design']})
non_numeric_course_mapped = pd.DataFrame({'Course': pd.Series([None, None, None], dtype='object')})
df_with_null_values_mapped = pd.DataFrame({'Course': ['Biofuel Production Technologies', None, 'Animation and Multimedia Design']})


