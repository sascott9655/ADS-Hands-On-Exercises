# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:22:38 2026

@author: samsc
"""

#Data Science Using Python and R: Chapter 8 - Page 126: Questions #31, 32, 33, & 34 


import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import statsmodels.tools.tools as stattools
from sklearn.metrics import confusion_matrix


fram_tr = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/framingham_nb_training.csv')
fram_tr
fram_tst = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/framingham_nb_test.csv')
fram_tst
#31. Run the Naive Bayes classifier to classify persons as living or dead based on sex and education.

t1 = pd.crosstab(
    fram_tr['Sex'], fram_tr['Educ'])
t1['Total'] = t1.sum(axis=1)
t1.loc['Total'] = t1.sum()
t1

t1_plot = pd.crosstab(fram_tr['Educ'], fram_tr['Sex']) 
t1_plot.plot(kind='bar', stacked=True)   

X = fram_tr[['Sex', 'Educ']]
Y = fram_tr['Death']

nb_01 = MultinomialNB().fit(X, Y)

#32. Evaluate the Naive Bayes model on the framingham_nb_test data set. Display the
# results in a contingency table. Edit the row and column names of the table to 
# make the table more readable. Include a total row and column. 

X_test = fram_tst[['Sex', 'Educ']]
y_predicted = nb_01.predict(X_test)

ypred = pd.crosstab(
    fram_tst['Death'], y_predicted, rownames=['Actual'], colnames=['Predicted'])
ypred['Total'] = ypred.sum(axis=1)
ypred.loc['Total'] = ypred.sum()
ypred


#33. According to your table in the previous exercise, find the following values for the Naive 
#Bayes model
#a.Accuracy
#b.Error rate

cm = confusion_matrix(fram_tst['Death'], y_predicted)
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy
error_rate= 1 -accuracy
error_rate

#34.According to your contingency table, find the following values for the Naive
#Bayes model:
#a. How often it correctly classifies dead persons
#It correctly classifies dead people about 48 percent of the time
#b. How often it correctly classifies living persons. 
#It does not correctly classify living people at all it just classifies if there are
#dead which it only corrects have the time

