import pandas as pd
import pytest
import sys
import os

# Import the map_int_categories function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.map_int_category import map_int_categories

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


# Test cases
def test_map_int_categories_valid_mapping():
    pd.testing.assert_frame_equal(map_int_categories(df_with_valid_course_codes, 'Course', course_mapping), df_with_valid_course_codes_mapped)

def test_map_int_categories_invalid_mapping():
    pd.testing.assert_frame_equal(map_int_categories(df_with_invalid_course_codes, 'Course', course_mapping), df_with_invalid_course_codes_mapped)

def test_map_int_categories_empty_df():
    pd.testing.assert_frame_equal(map_int_categories(df_empty, 'Course', course_mapping), df_empty_mapped)

def test_map_int_categories_type_errors():
    with pytest.raises(TypeError):
        map_int_categories(df_with_valid_course_codes, 'Course', ['Not', 'a', 'dict'])
    with pytest.raises(TypeError):
        map_int_categories('Not a DataFrame', 'Course', course_mapping)
    with pytest.raises(TypeError):
        map_int_categories(df_with_valid_course_codes, 123, course_mapping)

def test_map_int_categories_non_existing_column():
    with pytest.raises(ValueError):
        map_int_categories(df_with_valid_course_codes, 'NonExistingColumn', course_mapping)

def test_map_int_categories_mixed_valid_invalid():
    pd.testing.assert_frame_equal(map_int_categories(mixed_course_codes_df, 'Course', course_mapping), mixed_course_codes_mapped)

def test_map_int_categories_non_integer_keys():
    with pytest.raises(ValueError):
        map_int_categories(df_with_valid_course_codes, 'Course', { '33': 'Invalid' })

def test_map_int_categories_non_string_values():
    with pytest.raises(ValueError):
        map_int_categories(df_with_valid_course_codes, 'Course', { 33: 171 })

def test_map_int_categories_non_numeric_course_column():
    with pytest.raises(ValueError):
        map_int_categories(non_numeric_course_df, 'Course', course_mapping)

def test_map_int_categories_with_null_values():
    with pytest.raises(ValueError):
        map_int_categories(df_with_null_values, 'Course', course_mapping)
