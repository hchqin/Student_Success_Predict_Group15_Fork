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
