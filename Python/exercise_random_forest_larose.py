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


adult_train = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/adult_ch6_training.csv')
adult_test = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/adult_ch6_test.csv')

#19. Use random forests on training data set to predict income using marital status
#capital gains and losses

y_train = adult_train['Income'] #dataframe with double brackets a series with single bracket

#random forest needs categorical variables to be dummy variables in order for model to run

mar_dummies_train = pd.get_dummies(adult_train['Marital status']).astype(int)
mar_dummies_train_pd = pd.DataFrame(mar_dummies_train)
X_train = pd.concat((adult_train[['Cap_Gains_Losses']], mar_dummies_train_pd), axis=1)

mar_dummies_test = pd.get_dummies(adult_test['Marital status']).astype(int)
mar_dummies_test_pd = pd.DataFrame(mar_dummies_test)
X_test = pd.concat((adult_test[['Cap_Gains_Losses']], mar_dummies_test_pd), axis=1)
y_test = adult_test['Income']

X_names = ['Cap_Gains_Losses', 'Divorced', 'Married', 'Separated', 'Widowed']

y_names = ['<=50K', '>50K']

rfy = np.ravel(y_train)

rf = RandomForestClassifier(n_estimators= 100, criterion='gini').fit(X_train, rfy)


#20. Use random forests on test data set that utilizes the same target and 
#predictor variables. Does the test data result match the training data result?

#Copilot help me build accuracy variables for predictions. 
# Training accuracy
rf_acc_train = (rf.predict(X_train) == y_train).mean()
print(rf_acc_train)
# Test accuracy
rf_acc_test = (rf.predict(X_test) == y_test).mean()
print(rf_acc_test)