# split_n_preprocess.py
# author: Yili Tang
# date: 2023-12-1

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import set_config


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--drop-column', type=str, help="Optional: columns to drop")
@click.option('--numeric-column', type=str, help="Optional: numeric columns")
@click.option('--categorical-column', type=str, help="Optional: categorical columns")
@click.option('--ordinal-column', type=str, help="Optional: ordinal columns")
@click.option('--binary-column', type=str, help="Optional: binary columns")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(raw_data, data_to, preprocessor_to, drop_column, numeric_column, categorical_column, ordinal_column, binary_column, seed):
    '''...'''
    
    set_config(transform_output="pandas")

    student_df = pd.read_csv(raw_data, sep=';')
    train_df, test_df = train_test_split(student_df, test_size=0.2, random_state = 123)

    status_mapping = {
    1: 'single',
    2: 'married',
    3: 'widower',
    4: 'divorced',
    5: 'facto union',
    6: 'legally separated'
    }

    train_df['Marital status'] = train_df['Marital status'].map(status_mapping)
    test_df['Marital status'] = test_df['Marital status'].map(status_mapping)

    course_mapping = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (evening attendance)'
    }

    train_df['Course'] = train_df['Course'].map(course_mapping)
    test_df['Course'] = test_df['Course'].map(course_mapping)

    course_mapping = {
        0: 'evening',
        1: 'daytime',
    }


    train_df['Daytime/evening attendance\t'] = train_df['Daytime/evening attendance\t'].map(course_mapping)
    test_df['Daytime/evening attendance\t'] = test_df['Daytime/evening attendance\t'].map(course_mapping)
    train_df.rename(columns={'Daytime/evening attendance\t': 'Daytime evening attendance'}, inplace=True)
    test_df.rename(columns={'Daytime/evening attendance\t': 'Daytime evening attendance'}, inplace=True)

    nation_mapping = {
        1: 'Portuguese',
        2: 'German',
        6: 'Spanish',
        11: 'Italian',
        13: 'Dutch',
        14: 'English',
        17: 'Lithuanian',
        21: 'Angolan',
        22: 'Cape Verdean',
        24: 'Guinean',
        25: 'Mozambican',
        26: 'Santomean',
        32: 'Turkish',
        41: 'Brazilian',
        62: 'Romanian',
        100: 'Moldova (Republic of)',
        101: 'Mexican',
        103: 'Ukrainian',
        105: 'Russian',
        108: 'Cuban',
        109: 'Colombian'
    }

    # Apply the mapping
    train_df['Nacionality'] = train_df['Nacionality'].map(nation_mapping)
    test_df['Nacionality'] = test_df['Nacionality'].map(nation_mapping)
    

    train_df.to_csv(os.path.join(data_to, "student_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_to, "student_test.csv"), index=False)

    target = "Target"

    if drop_column:
        drop_features = pd.read_csv(drop_column, header=None).loc[:,0].tolist()
    
    if numeric_column:
        numeric_features = pd.read_csv(numeric_column, header=None).loc[:,0].tolist()

    if categorical_column:
        categorical_features = pd.read_csv(categorical_column, header=None).loc[:,0].tolist()
    
    if ordinal_column:
        ordinal_features = pd.read_csv(ordinal_column, header=None).loc[:,0].tolist()
    
    if binary_column:
        binary_features = pd.read_csv(binary_column, header=None).loc[:,0].tolist()

    X_train = train_df.drop(columns=["Target"])
    X_test = test_df.drop(columns=["Target"])
    y_train = train_df["Target"]
    y_test = test_df["Target"]

    ordinal_transformer = OrdinalEncoder(dtype=np.int64, handle_unknown='use_encoded_value', unknown_value=-1)
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int, min_frequency=10)
    numeric_transformer = StandardScaler()

    student_preprocessor = make_column_transformer(
        ( numeric_transformer, numeric_features),  
        ( categorical_transformer, categorical_features+binary_features),  
        ( ordinal_transformer, ordinal_features),
        ("drop", drop_features),
        verbose_feature_names_out=False
    )
    
    pickle.dump(student_preprocessor, open(os.path.join(preprocessor_to, "student_preprocessor.pickle"), "wb"))

    student_preprocessor.fit(X_train)
    scaled_student_train = student_preprocessor.transform(X_train)
    scaled_student_test = student_preprocessor.transform(X_test)

    scaled_student_train.to_csv(os.path.join(data_to, "scaled_student_train.csv"), index=False)
    scaled_student_test.to_csv(os.path.join(data_to, "scaled_student_test.csv"), index=False)

if __name__ == '__main__':
    main()
