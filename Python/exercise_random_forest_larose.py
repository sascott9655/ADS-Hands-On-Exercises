# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:16:55 2026

@author: samsc
"""

#Data Science Using Python and R: Chapter 6 - Page 93: Questions #19 & 20

#Use both R and Python for these questions.

import pandas as pd
import numpy as np
import statsmodels.tools.tools as stattools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


adult_train = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/adult_ch6_training.csv')
adult_test = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/adult_ch6_test.csv')

#19. Use random forests on training data set to predict income using marital status
#capital gains and losses

y = adult_train[['Income']] #dataframe with double brackets a series with single bracket
adult_train['Marital status'].value_counts()
#random forest needs categorical variables to be dummy variables in order for model to run

mar_dummies = pd.get_dummies(adult_train['Marital status']).astype(int)
mar_dummies_pd = pd.DataFrame(mar_dummies)
X = pd.concat((adult_train[['Cap_Gains_Losses']], mar_dummies_pd), axis=1)

X_names = ['Cap_Gains_Losses', 'Divorced', 'Married', 'Separated', 'Widowed']

y_names = ['<=50K', '>50K']

rfy = np.ravel(y)

rf = RandomForestClassifier(n_estimators= 100, criterion='gini').fit(X, rfy)

train_model = rf.predict(X)

#20. Use random forests on test data set that utilizes the same target and 
#predictor variables. Does the test data result match the training data result?

y2 = adult_test[['Income']] #dataframe with double brackets a series with single bracket
adult_test['Marital status'].value_counts()
#random forest needs categorical variables to be dummy variables in order for model to run

mar_dummies2 = pd.get_dummies(adult_test['Marital status']).astype(int)
mar_dummies_pd2 = pd.DataFrame(mar_dummies2)
X2 = pd.concat((adult_test[['Cap_Gains_Losses']], mar_dummies_pd2), axis=1)

X_names2 = ['Cap_Gains_Losses', 'Divorced', 'Married', 'Separated', 'Widowed']

y_names2 = ['<=50K', '>50K']

rfy2 = np.ravel(y2)

rf2 = RandomForestClassifier(n_estimators= 100, criterion='gini').fit(X2, rfy2)

test_model = rf2.predict(X2)


train_acc = accuracy_score(rfy, train_model)
test_acc = accuracy_score(rfy2, test_model)

train_acc, test_acc







