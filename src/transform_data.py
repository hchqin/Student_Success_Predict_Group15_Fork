import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

def transform_data(X_train, preprocessor):
    """
    Transforms the training data using the provided preprocessor and returns it as a pandas DataFrame.

    Parameters:
    X_train (pd.DataFrame): The raw training data.
    preprocessor (ColumnTransformer): The preprocessor to be applied to the data.

    Returns:
    pd.DataFrame: Transformed training data.

    Raises:
    ValueError: If the input data is not a pandas DataFrame.
    NotFittedError: If the preprocessor has not been fitted.
    Exception: For other unexpected errors during transformation.
    """

    if not isinstance(X_train, pd.DataFrame):
        raise ValueError("Input X_train must be a pandas DataFrame")

    try:
        # Transform the data
        transformed = preprocessor.fit_transform(X_train)

        # Get feature names
        col_names = preprocessor.get_feature_names_out().tolist()

        # Create and return a DataFrame with the transformed data
        return pd.DataFrame(transformed, columns=col_names)

    except NotFittedError as nfe:
        raise NotFittedError(f"Preprocessor not fitted: {nfe}")

    except Exception as e:
        raise Exception(f"Error in data transformation: {e}")
