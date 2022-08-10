#!/bin/sh
# # Import relevant libraries
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
4. Model experimentation and evaluation

- This is a binary classification machine learning model to predict if the outcome is a show or no show. 
- The datset is moderately imbalanced and we will need to adddress this issue by 1) adjusting class weights 2) Using SMOTE to balance the dataset.

The dataset will first be split into training and test sets. The training set will be used to train the model and the test set will be used to evaluate the model. Due to the imbalance of the dataset, we will use stratified train test split to ensure that the training and test sets are balanced.

In addition, oversampling techniques such as SMOTE and class weights adjustment will be used. As this is a moderately imbalanced dataset, these techniques will be applied after the train-test split of the dataset and only on the train set. This is to prevent:

Data Leakage and idealistic situation of the reality where class labels are balanced.  Therefore, the test dataset will not be oversampled to reflect the reality and allow the model to show its performance in a context that is close to the reality.

Other than using oversampling techniques, we will also explore the method fo adjusting class weights to penalise the minority class labels during model training. This can be another way to deal with this set of imbalanced dataset.

The ML algorithms that will be used are:
1. Logistic Regression 
2. K Nearest Neighbours
3. Naives Bayes
4. Random Forest
5. XGBoost


Evaluation Metrics: 

1. F-1 Score - It is calculated as the harmonic mean of precision and recall. It is used as an evaluation metric as it balances precision and recall of the classes. 
2. AUC - Area Under Curve - It is calculated as the area under the ROC curve
3. Precision - Recall Curve - It is calculated as the area under the precision recall curve

"""

"""
(a) Using SMOTE to balance the dataset
(b) Using Class weights adjustment during model training

The function fit_score() will be created to perform preliminary training using baseline models:

1. train_test_split
2. Fit and predict each defined baseline model by running through a loop using pipeline
3. Stratified Cross Validation on train set
4. Print Classification Report , AUC Score and mean CV score for each model

The best performing model will then be selected for hyper-parameter tuning. 
"""

# Create the function fit_score to streamline the training process and provide relevant metrics for evaluation.
"""
Parameters: 
X - Dataframe of features
y - Dataframe of labels
imbal_tech - Oversampling technique 
preprocessor - Preprocessing technique (MinMaxScaler, StandardScaler, OrdinalEncoder)
classifiers - Dictionary of classifiers
imbalance - Boolean to determine if oversampling technique is applied
"""
# In the event oversampling is done, set imbalance = True , if not, set imbalance = False 
def fit_score (X, y, imbal_tech, preprocessor, classifiers, imbalance = True):
    # Perform train test split on data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size= 0.25,
                                                    stratify= y,random_state= 111)
    
    # Create empty lists for AUC score and cross validation scores
    auc = {}
    CV = {}
    # Run a loop to go through every classifier model    
    for name, classifier in classifiers.items():
        if imbalance == True:     # if oversampling technique is applied
            pipe = Pipeline(steps=[('imbal_tech', imbal_tech), ('preprocessor', preprocessor),
                      ('classifier', classifier)])
           
        else:   # if oversampling method is not applied 
            pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
            
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
   
       
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle = True,random_state= random)   # Do stratified kfold for CV
    
        scores = cross_val_score(pipe, X_train, y_train, cv = stratified_kfold, scoring = 'f1')
        print(name, '\n')
        print(classification_report(y_test, y_pred))
        auc[name] = roc_auc_score(y, classifier.predict_proba(X)[:, 1])
        CV [name] = np.mean(scores) 
    
    return (print('AUC Score: \n', auc, '\n'),  print('Mean CV Score (F1): \n', CV))

