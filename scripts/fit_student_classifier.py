# fit_student_classifier.py
# author: Bill Wan
# date: 2023-12-2

import click
import os
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump



@click.command()
@click.option('--original-train', type=str, help="Path to original training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the model pipeline object will be written to")
@click.option('--result-to', type=str, help="Path to directory where the result will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(original_train, preprocessor, pipeline_to, result_to, seed):
    '''...'''

    np.random.seed(seed)
    set_config(transform_output="pandas")

    train_df = pd.read_csv(original_train)
    X_train = train_df.drop(columns=["Target"])
    y_train = train_df["Target"]
    student_preprocessor = pickle.load(open(preprocessor, "rb"))

    param_dist_rf = {
        "randomforestclassifier__n_estimators": randint(50, 200),   
        "randomforestclassifier__max_depth": randint(5, 15),
        "randomforestclassifier__min_samples_split": randint(5, 10),  
        "randomforestclassifier__min_samples_leaf": randint(5, 10), 
        
    }

    rf_pipe = make_pipeline(student_preprocessor, RandomForestClassifier(random_state=123))
    rf_random_search = RandomizedSearchCV(
        rf_pipe, param_dist_rf, n_iter=100, n_jobs=-1, return_train_score=True, random_state=123
    )
    rf_fit = rf_random_search.fit(X_train, y_train)

    with open(os.path.join(pipeline_to, "RF_model.pickle"), 'wb') as f:
        pickle.dump(rf_fit, f)


    param_dist_lr = {
    'logisticregression__C': uniform(0.01, 100.0), 
    'logisticregression__penalty': ['l2', 'l1'] 
    }

    lr_pipe = make_pipeline(student_preprocessor, LogisticRegression(max_iter=2000, random_state=123, solver='liblinear', multi_class="ovr"))
    lr_random_search = RandomizedSearchCV(
        lr_pipe, param_dist_lr, n_iter=100, n_jobs=-1, cv=5, random_state=123
    )
    lr_fit = lr_random_search.fit(X_train, y_train)
    with open(os.path.join(pipeline_to, "LR_model.pickle"), 'wb') as f:
        pickle.dump(lr_fit, f)

    param_dist_svc = {"svc__C": loguniform(1e-3, 1e1), 
                      "svc__gamma": loguniform(1e-3, 1e1)}

    svc_pipe = make_pipeline(student_preprocessor, SVC(probability=True, random_state=123)) 

    svc_random_search = RandomizedSearchCV(
        svc_pipe, param_dist_svc, n_iter=100, n_jobs=-1, cv=5, random_state=123)
    
    svc_fit = svc_random_search.fit(X_train, y_train)

    with open(os.path.join(pipeline_to, "SVC_model.pickle"), 'wb') as f:
        pickle.dump(svc_fit, f)

    models = {
        "RandomForest": rf_fit.best_estimator_,
        "Logistic Regression": lr_fit.best_estimator_,
        "SVC": svc_random_search.best_estimator_
    }

    score_types_class = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average='weighted'),
        "recall": make_scorer(recall_score, average='weighted'),
        "f1": make_scorer(f1_score, average='weighted'),
        "roc_auc": "roc_auc_ovr"
    }

    cv_results = {}
    for model_name, model in models.items():
        scores = cross_validate(model, X_train, y_train, scoring=score_types_class, cv=5, return_train_score=True)
        cv_results[model_name] = pd.DataFrame(scores).mean()

    results_df = pd.DataFrame(cv_results)
    results_df.to_csv(os.path.join(result_to, "training_scores_cv.csv"), index=True)

if __name__ == '__main__':
    main()