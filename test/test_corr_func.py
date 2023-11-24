import pandas as pd
import pytest
import sys
import os
import numpy as np

# Import the get_high_correlations function from the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.high_corr_extract import get_high_correlations  