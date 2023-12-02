# eda.py
# author: Yili Tang
# date: 2023-12-2

import click
import os
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@click.command()
@click.option('--training-data', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")

def main(training_data, plot_to):
    '''...'''

    train_df = pd.read_csv(training_data)
    categorical_features = ['Marital status', 'Application mode', 'Course', 'Nacionality',  "Mother's occupation", "Father's occupation"]
    binary_features = ['Daytime evening attendance', 'Displaced', 'Educational special needs', 
                       'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
    numeric_features = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment', 
       'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 
       'Unemployment rate', 'Inflation rate',  'GDP']


    n_features = len(categorical_features[:3] + binary_features)
    n_rows = (n_features + 1) // 2

    if n_features > 0:
        fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows))

        for i, feat in enumerate(categorical_features[:3] + binary_features):
            row = i // 2
            col = i % 2
            sns.countplot(data=train_df, x=feat, hue="Target", palette="Set2", alpha=0.6, ax=axes[row, col])
            axes[row, col].set_title("Distribution of " + feat + " by Target", fontweight='bold')
            axes[row, col].set_ylabel("Count", fontweight='bold')
            axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), rotation=45, ha='right')

    if n_features % 2 == 1:
        fig.delaxes(axes[-1, -1])

    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "distribution_of_categorical_feature.png"), dpi=300, bbox_inches='tight')


    n_features = len(numeric_features)
    n_rows = (n_features + 1) // 2  

    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 6 * n_rows))

    for i, feat in enumerate(numeric_features):
        row = i // 2
        col = i % 2
        sns.histplot(data=train_df, x=feat, hue="Target", kde=True, palette="Set2", element="step", ax=axes[row, col])
        axes[row, col].set_title("Distribution of " + feat + " by Target", fontweight='bold')
        axes[row, col].set_xlabel(feat, fontweight='bold')
        axes[row, col].set_ylabel("Density", fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "density_of_numeric_feature.png"), dpi=300, bbox_inches='tight')


    numeric_subset = train_df.loc[:, numeric_features]

    corr_matrix = numeric_subset.corr(method='spearman')

    plt.figure(figsize=(16, 12))  
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='viridis')

    plt.savefig(os.path.join(plot_to, "heat_map.png"), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()