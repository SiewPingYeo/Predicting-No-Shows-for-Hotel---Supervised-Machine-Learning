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
1. Load data from database using sqlite
"""

# Create a function to extract data from db and convert it to pandas df 
"""
Parameters:
    db_name: name of the database
    query: sql query to extract data from database
"""
def connect_db(db_name, query):
    # Set up connection to database using sqlite3
    conn = sqlite3.connect(f'data/{db_name}.db')
    df = pd.read_sql(query,conn)
    #close the connection
    conn.close() 
    return df


"""
2. Preprocess data
"""

#Create a function to drop rows with all null values and duplicates 
def basic_clean(df):
    print('Cleaning data...')
    df.dropna(axis = 0, how = 'all', inplace = True)

    # Remove duplicates 
    df.drop_duplicates(keep= 'first', inplace = True)
    
    # Reset index of the dataset after dropping row
    df.reset_index(drop= True, inplace = True)

    print('Data cleaned!')
    return df


# Clean up price column - do currency conversion and remove all the dollar signs and commas
# Using usd $1 = sgd $1.39 accurate as of 16th June 2022
# Create a function to convert currency
def price_convert (x):
    if x is not np.nan:
        x = str(x).split('$')
        if x[0] == 'USD':
            x[1] = float(x[1])*1.39
        else:
            x[1] = float(x[1])
    
    return x[1]


# Create a function to clean up the price column
# Replace all null values with SGD$ 0 for price column  and replace with mean later on
def price_clean(df):
    print('Cleaning price data...')
    df.loc[df['price'].isnull(), 'price'] = 'SGD$ 0'
    # Apply function to standardise the price in SGD for price column 
    df['price'] = df['price'].apply(price_convert)
    # Replace price = 0.00 with mean for price column 
    df.loc[df['price'] == 0, 'price'] = df['price'].mean()
    print('Price data cleaned!')
    return df


# Create a function to further clean up the columns
def clean_col(df):
    print('Cleaning column data...')
    # Fill in missing values for room column using Mode
    df['room'] = df['room'].fillna(df['room'].mode()[0])
    
    # Change datatype to integer for no_show column 
    df['no_show'] = df['no_show'].astype(int)

    # Convert all the months to lowercase 
    df['arrival_month'] = df['arrival_month'].str.lower()

    # Convert datatype to integer for arrival_day column
    df['arrival_day'] = df['arrival_day'].astype(int)
    df['arrival_day'] = df['arrival_day'].astype(str)

    # Mod all values to remove the minus sign for checkout_day as it might be due to typo 
    df['checkout_day'] = df['checkout_day'].abs()
    df['checkout_day'] = df['checkout_day'].astype(int)
    df['checkout_day'] = df['checkout_day'].astype(str)

    # Replace 'one' with 1 and 'two' with 2 for num_adults column
    df.loc[df['num_adults']== 'one', 'num_adults'] = 1
    df.loc[df['num_adults']== 'two', 'num_adults'] = 2

    # Convert datatype to str for num_adults column
    df['num_adults'] = df['num_adults'].astype(str)

    # Convert datatype to string for num_children column
    df['num_children'] = df['num_children'].astype(int)
    df['num_children'] = df['num_children'].astype(str)

    # Log transform price as it is slightly right-skewed 
    df['price'] = np.log((1+ df['price']))
    print('Column data cleaned!')
    return df 

"""
3. Feature Engineering
"""
# Create a function to pre-process the data by doing get_dummies and frequency encoding 

"""
Parameters:
    df: dataframe to be pre-processed
    dummies : list of columns to be converted to dummies
    freq_encodings: list of columns to be frequency encoded

"""
def data_prep(dummies, df, freqlist):
    print('Preprocessing data...')
    dum = pd.get_dummies(df[dummies], drop_first=True, dtype= 'int64')
    df1 = pd.concat([df, dum], axis = 1)
    df1.drop(dummies, axis = 1, inplace = True)

    df_fe = pd.DataFrame()
    for i in freqlist:
        counts = df1[i].value_counts()
    # get frequency of each category
        encoding = counts/len(df1)
    # map() maps the frequency of each category to each observation
        df_fe[i] = df1[i].map(encoding)
        df_fe[i] = df_fe[i].astype(float)
    df1.drop(freqlist, axis = 1, inplace = True)  # Drop original columns 
    # Concat the results to the original dataframe 
    df1= pd.concat([df1, df_fe], axis = 1)
    print('Data preprocessed!')
    return(df1)
