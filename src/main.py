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


from ml_module.data_preprocessing import connect_db, basic_clean, price_clean, clean_col, data_prep
from ml_module.model_exp import fit_score
from ml_module.hyperparameter_tuning import hyperparam_tuning

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
1. Load data from database using sqlite
"""

# sql query to select all columns from the table noshow
db_name = 'noshow'
query = "Select * from noshow;"

#Apply function
df = connect_db (db_name, query)


"""
2. Preprocess data
"""

df = basic_clean(df)

# Clean up price column - do currency conversion and remove all the dollar signs and commas
# Using usd $1 = sgd $1.39 accurate as of 16th June 2022

df = price_clean(df)

df = clean_col(df)

"""
3. Feature Engineering
"""
# Create a list containing the columns to do get_dummies encoding
dummies= ['branch', 'country', 'first_time', 'room', 'platform', 'num_adults', 'num_children']
# Create a list containing the columns to do frequency encoding 
freqlist = ['booking_month', 'arrival_month', 'arrival_day', 'checkout_month', 'checkout_day']


df = data_prep(dummies, df, freqlist)


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

# Define X and y dataset for Machine Learning
X = df.iloc[:, 1::] 
y = df.iloc[:, 0]

"""
(a) Using SMOTE to balance the dataset

The function fit_score() will be created to perform preliminary training using baseline models:

1. train_test_split
2. Fit and predict defined baseline models
3. Stratified Cross Validation on train set
4. Print Classification Report , AUC Score and mean CV score for each model
"""

# Define the oversampling technique to be applied 
imbal_tech = SMOTE(random_state= random)

# Define the scaling or normalisation technique to be applied
preprocessor =  StandardScaler()

# Define the classifiers to be used for model training
classifiers = {"Logistic Regression": LogisticRegression(solver = 'liblinear', random_state = random),
            "KNN": KNeighborsClassifier(n_neighbors = 5, p = 2),    
          "Naives Bayes": GaussianNB(),
          "Random Forest Classifier": RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = random),
          "XGboost": XGBClassifier(objective='binary:logistic', random_state = random) 
         }


fit_score(X, y, imbal_tech, preprocessor, classifiers, imbalance = True)

"""
(b) Using Class weights adjustment during model training 
"""
# No oversampling technique is needed, thus we assign it to an empty list
imbal_tech = []

# Define the scaling or normalisation technique to be applied
preprocessor =  StandardScaler()

# Calculate the weight ratio of class 0 and 1 
weight_ratio = len(df.loc[df['no_show']== 0])/len(df.loc[df['no_show']== 1])

# Define the classifiers to be used for model training
classifiers = {"Logistic Regression": LogisticRegression(solver = 'liblinear', class_weight = 'balanced', random_state = random),
            "KNN": KNeighborsClassifier(n_neighbors = 5, p = 2, weights = 'distance'),
          "Random Forest Classifier": RandomForestClassifier(n_estimators = 50, criterion = 'entropy', class_weight = 'balanced', random_state = random),
          "XGboost": XGBClassifier(scale_pos_weight= weight_ratio , objective='binary:logistic', random_state =random)}


# set imbalance = False as there are no oversampling techniques to be applied
fit_score (X, y, imbal_tech, preprocessor, classifiers, imbalance = False)

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

# define models and parameters
# Calculate the weight ratio of class 0 and 1 
weight_ratio = len(df.loc[df['no_show']== 0])/len(df.loc[df['no_show']== 1])
model = XGBClassifier(scale_pos_weight= weight_ratio , objective='binary:logistic', random_state =random, eval_metric = 'error')

eta = [0.05, 0.1,0.15, 0.2, 0.3]
gamma = [0, 0.05, 0.5, 0.8, 1.0]
max_depth = [3,6,7,8, 9, 10]
min_child_weight = [1,2,3,4,5]
subsample = [0.6,0.7,0.8,0.9,1.0] 

# define grid search
param_grid = dict(eta=eta,gamma=gamma,max_depth= max_depth, min_child_weight= min_child_weight, subsample= subsample)
# number of folds for cross validation
cv = 3 
# define scoring metric
scoring = 'f1'

model, X_train, X_test, y_train, y_test = hyperparam_tuning(X, y, model, param_grid, cv, scoring)


# Which are the top features used in the model?
#plot_importance(model)
#plt.show()

#The feature importance plot shows that price , checkout_day, arrival_day and booking_month are the top most important features for the model.

# Pickle model for future usage 
with open('model.pkl', 'wb') as files:
    pickle.dump(model, files)