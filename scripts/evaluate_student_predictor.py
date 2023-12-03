# evaluate_student_predictor.py
# author: Yili Tang
# date: 2023-12-2

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn import set_config

@click.command()
@click.option('--original-test', type=str, help="Path to original testing data")
@click.option('--scaled-test-data', type=str, help="Path to scaled test data")
@click.option('--rf-from', type=str, help="Path to directory where the fit rf object lives")
@click.option('--lr-from', type=str, help="Path to directory where the fit lr object lives")
@click.option('--svc-from', type=str, help="Path to directory where the fit svc object lives")
@click.option('--results-to', type=str, help="Path to directory where the results will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(original_test, scaled_test_data, rf_from, lr_from, svc_from, results_to, seed):
    '''Main function to evaluate machine learning models on student data.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    test_df = pd.read_csv(original_test)
    X_test = test_df.drop(columns=['Target'])
    y_test = test_df['Target']
    
    models = {}
    for model_file, model_name in zip([rf_from, lr_from, svc_from], ["RandomForest", "LogisticRegression", "SVC"]):
        with open(model_file, 'rb') as f:
            models[model_name] = pickle.load(f).best_estimator_

    test_results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        test_results[model_name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr")
        }

    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(os.path.join(results_to, "test_scores.csv"), index=True)

if __name__ == '__main__':
    main()
