#!/bin/sh
# Import relevant libraries
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')

import pickle
#import warnings

#import scipy.stats as spstats
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from numpy import arange, argmax
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ( RandomizedSearchCV,
                                     RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, plot_importance

random = 42

""" 
Workflow:

1. Load data from database using sqlite
2. Preprocess data
3. Feature Engineering 
4. Model experimentation and evaluation
5. Model selection and hyperparameter tuning
6. Train model with best hyperparameters and pickle model for future use

"""

"""
5. Model selection and hyperparameter tuning

Out of the models that were trained using the baseline models, the random forest model and XGBoost model produces a higher f1-score for class 1 (No-show) for both SMOTE and class weights adjustment. 
However, for class weights adjustment, the XG Boost model has a higher AUC score at 0.692 compared to that of the random forest model at 0.687. 
Therefore, the XGBoost model will be chosen for hyperparameter tuning. 

Algorithms like Decision Tree and Ensemble Techniques (like AdaBoost and XGBoost) do not require scaling because splitting in these cases are based on the values. 
Therefore, standard scaler will not be used for XGBoost hyperparameter tuning.


6. Train model with best hyperparameters and pickle model for future use

Hyper parameter tuning was done using RandomizedSearchCV and the f1 score and AUC score improved slightly compared to the baseline model. The f1 score for class 1 is now at 0.65 and the AUC score is at 0.80, which is pretty good.
We will now construct the XGBoost model using the best parameters and use it for prediction. 
"""
# Create a function to run hyperparameter tuning for XGBoost , this function can be applied to any model that requires hyperparameter tuning
"""
Parameters:

X: Dataframe containing the features
y: Dataframe containing the target variable
model: can be a list of models or a single model
param_grid: dictionary containing the parameters to be tuned
cv: cross validation strategy
scoring: scoring metric to be used for hyperparameter tuning
"""
def hyperparam_tuning (X, y, model, param_grid, cv , scoring ):

    # Perform train test split on data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size= 0.25,
                                                    stratify= y,random_state= 111)

    #cv = StratifiedKFold(n_splits=10, shuffle = True, random_state= random)
    rand_search = RandomizedSearchCV(estimator=model,param_distributions= param_grid, n_jobs=-1, cv=cv ,scoring = 'f1',error_score=0, verbose = 1, random_state = random)
    rand_result = rand_search.fit(X_train, y_train)
    y_pred = rand_search.predict(X_test)
    # Print classification report 
    print(classification_report(y_test, y_pred))
    #Best parameters from grid search
    print(rand_search.best_params_)
    # Print AUC score 
    y_pred_proba = rand_search.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    print(auc)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()

    # Plot the model precision-recall curve to further evaluate how the model is doing at different probability thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    # plot the model precision-recall curve
    plt.plot(recall, precision, marker='.', label='XGBoost')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.title('Precision-Recall Curve')
    plt.show()

    return rand_search, X_train, X_test, y_train, y_test


