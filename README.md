# Predicting-No-Shows-for-Hotel---Supervised-Machine-Learning Pipeline

## Overview of folder structure 
![folder structure](https://user-images.githubusercontent.com/94337686/174553326-c170127f-c978-4445-a31f-ebedcd1f8681.jpg)



## Instructions on file usage 
1. For exploratory data analysis -  Refer to EDA.ipynb 
2. For ML pipeline - Refer to .src folder
    - main.py will run the the end-to-end pipeline
    
    Under ml_modules: 
    - data_preprocessing.py stores the functions needed to clean and pre-process the data (connect_db(), basic_clean(), price_clean(), clean_col(), data_prep())
    - model_exp.py stores the functions needed to run model training iterations across selected models, preprocessing tools and techniques to address imbalanced dataset as well as the evaluation metrics needed (fit_score())
    - hyperparameter_tuning.py stores the functions needed to run hyper-parameter tuning for the selected model and produce necessary evaluation metrics (hyperparam_tuning())
    
      **All parameters to be adjusted for each function are further described in the individual .py scripts** 
3. For data used in this project - Refer to data folder 
4. Refer to requirements.txt for the dependencies used in this entire project 
5. Refer to run.sh for the execution of the .py files 

#### To execute the ML pipeline, run the run.sh file to execute the main.py script which will then extract the required functions in the .py scripts under folder ml_modules. 

## Problem Statement 
Aim: To predict the No-Show of customers (using the dataset provided) to help a hotel chain
to formulate policies to reduce expenses incurred due to No-Shows. As there are only two outcomes, this will be a supervised binary classification model. 

## Workflow of this project

**(a) Exploratory Data Analysis:**
  1. Import relevant libraries 
  2. Connect to database and import data using sqlite3
  3. Data Exploration and Cleaning 
  4. EDA - Univariate, Bivariate and Multivariate exploration 

**(b) Machine Learning Pipeline**
  1. Load data from database using sqlite3
  2. Preprocess data
        - Cleaning of columns and ensuring datatype is correct 
        - Log transformation of price column 
  3. Feature Engineering
        - Encoding of features - frequency encoding and get_dummies 
  5. Model experimentation and evaluation
  6. Model selection and hyperparameter tuning
  7. Train model with best hyperparameters and pickle model for future use

## (a) Exploratory Data Analysis 

To look into how the features interact with the label of no_show, univariate analysis is first done to sieve out interesting trends in relation to no show cases. Subsequently, a deep dive into the interesting insights obtained in the univariate analysis will be performed. 

**Some observations from univariate analysis :** 
1. Majority of the no show cases are from Changi hotel branch.
2. No shows are slightly higher in booking months of June, July and September
3. No shows are the highest during the arrival months of May to August and this seems to be in line with the summer season of the year.
4. No shows mainly falls within checkout months of April to October.
5. A significant number of no shows were visitors from China. 
6. Large majority of no shows are from visitors who booked the room at the hotel for the first time and from those who booked a king sized room 
7. Most of the no shows are from visitors who booked a room through websites or emails. Perhaps websites and emails are the most convenient way to book a hotel room and there are a large number of bookings through websites and emails, thus the no show rate through thes eplatforms might be higher as well. 
8. No shows tend to be slightly higher for 1 adult and those with 0 or 1 child.


**Insights from univariate analysis:** 
- It seems like there are more no show cases in the summer months such as May-August, irregardless of whether it is a booking month, arrival month or checkout month. These might be a period where it is the travelling peak during the summer holidays. Higher rates of booking/arrival/checkouts during this period may also naturally lead to higher probability of getting no show cases. 

- Most of the no show cases come from Changi Branch and there might be a few reasons which influence the no show rate.
    - The appeal of Changi branch might not be as strong, thus vistors who subsequently found better hotels might not be interested. 
    - Changi branch might impose lesser penalties/ less strict in cancellation and no show policies.
    - Exploration on the difference between these two branches can explored further in the next section. 


- Hypothesis - The lower the room price is , the higher the tendency for vistors to do a no show
    - Since most of the no shows are from visitors who booked a room through websites or emails, more can be explored on whether hotel prices differs between these platforms and cheaper hotel rooms might make it easier for visitors to forego the room and do a no show.

    - Since a large majority of no shows come from China, it might be a good idea to explore the differences in the room prices across countries to see if prices do affect the number of no shows. 

    - It will also be interesting to see how prices vary across room types. Logically, king-sized rooms are more expensive, but the no show cases for king sized rooms are the highest. 

    - How does price vary for a person who booked the first time and a person who booked a room multiple times? 

**Insights from bivariate and multivariate analysis:**

The insights gained from univariate analysis are further explored in this section. The basis of this section will be on the hypothesis that lower room prices lead to higher no show cases. The analysis will explore across the interactions of various features with price to see whether price actually affects the occurences of no show. 


![price across branches](https://user-images.githubusercontent.com/94337686/174483722-69161b58-3eb9-4dfc-a3c5-effc4e684cf7.png)



![platforms, country and price](https://user-images.githubusercontent.com/94337686/174483736-6fa5423f-1fc5-43aa-8851-ac46782371a6.png)


![prices, room types, platforms](https://user-images.githubusercontent.com/94337686/174483748-52fe5a35-ac7e-4b3b-a460-ef8c2cc3c6e3.png)


![first time](https://user-images.githubusercontent.com/94337686/174483759-6a891178-f1c4-4b94-a416-fac19aef2fc5.png)


This preliminary analysis shows that price does not play a huge role in the occurences of no show cases. There might be other factors at play as well. 



**Feature Engineering**

- EDA allows a better overview of the dataset for feature engineering. Get dummies and frequency encoding are performed for categorical features while log transformation is performed the continuous feature in order to prepare the data for model training later on. 

![log trans](https://user-images.githubusercontent.com/94337686/174483941-d859856c-db90-4e50-bb7d-85e508575b56.png)



## (b) Machine Learning Pipeline 

In order to prepare the data for machine learning, data pre-processing and feature engineering are performed in this section. The dataset is also moderately imbalanced, with the occurences of class 0 having around 1.71 times more than the occurences of class 1, as uncovered in the EDA section. Therefore, techniques to address the issue of imabalanced dataset has to be applied. Standard Scaling is also done for the dataset before being fed into the models. 

In this section, model iterations are ran in two different scenarios during the experimentation stage. 
1. Using SMOTE on the dataset (SMOTE is an oversampling technique where the synthetic samples are generated for the minority class.) 
2. Adjusting class weights during the model training process 

The ML algorithms explored in the experiementation stage are:
1. Logistic Regression 
2. K Nearest Neighbours
3. Naives Bayes
4. Random Forest
5. XGBoost

The algorithms are chosen as they are supervised classification models appropriate for this project which has binary outcomes of 0 or 1 and where data is labelled. For every iteration, cross validation is also done to ensure that the model does not overfit.  

**Evaluation metrics**

The evaluation metrics used in this project are f1-score and AUC score. 

1. F1-score - The F1-score combines the precision and recall of a classifier into a single metric by taking their harmonic mean.
      - Both precison and recall can be important in this case study. Recall is important as the hotel would want to catch as many noshows as they can to reduce revenue loss , while precision is essential to ensure that the hotel does not spend more than necessary to prevent no show cases. Therefore, the f1 score will give a combination of these two scores and serve as the main evaluation metric. The higher the F1-score, the better the model performance. 


2. AUC score - The Receiver Operator Characteristic (ROC) curve is an evaluation metric for binary classification problems. It is a probability curve that plots the TPR against FPR at various threshold values and essentially separates the ‘signal’ from the ‘noise’. The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve.

The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.
The AUC score will thus give a better overview of the model performance. 

In order to look at the evaluation metrics , the classification report and AUC score is used. 

**Best model selected for hyperparameter tuning**

Out of the models trained, the xgboost model with class weights adjusted is selected based on a combination of F1 score and AUC score. It has the highest F1-score and AUC score out of all.
Hyper-parameter tuning is then done for the xgboost model. The eta, gamma, max_depth, max_child_weight and subsample were tuned using RandomizedSearch CV. 
The F1 score and AUC score improved slightly after tuning. 

The classifictaion report after hyper-parameter tuning:

![Confusion matrix](https://user-images.githubusercontent.com/94337686/174483581-4762568d-6a90-45b7-adbf-d57f94d27e41.jpg)

The ROC-AUC score after hyper-parameter tuning:

![ROC-AUC](https://user-images.githubusercontent.com/94337686/174483604-c3012c15-4bf9-477a-a9dc-45ba69bc87fa.jpg)

The Precision-Recall curve is also plotted out to allow an overview of how the model balances precision and recall at different probability thresholds.

![precision recall curve](https://user-images.githubusercontent.com/94337686/174483634-b15a7175-cb81-4480-a2fa-080cfb059352.jpg)


The feature importance plot from xgboost shows that price , checkout_day, arrival_day and booking_month are the top most important features for the model. 
To improve on the model, one possible way for further exploration would be to drop the features of least importance and obtain more relevant features that could help in the model prediction. 

![feature importance](https://user-images.githubusercontent.com/94337686/174483644-72dce4b7-846d-4eab-87fd-e153a0ee5751.jpg)


Finally, the model is then pickled for future usage or deployment. 






