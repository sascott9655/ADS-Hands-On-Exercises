# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 10:13:04 2026

@author: samsc
"""

#Data Science Using Python and R: Chapter 13 - Page 195: Questions #13, 14, 15, 16, & 17

#Use both R and Python for these questions.

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

sales_train = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/clothing_sales_training.csv')
sales_test = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/clothing_sales_test.csv')


#13. Create a logistic regression model to predict whether or not a customer has a store
#credit card, based on whether they have a web account and the days between pruchases.
#Obtain the summary of the model

X = pd.DataFrame(sales_train[['Days', 'Web']])
X= sm.add_constant(X)
y = pd.DataFrame(sales_train['CC'])

logreg01 = sm.Logit(y, X).fit()
logreg01.summary2()


#14. Are there any variables that should be removed from the model? If so remove them 
# and rerun the model?

#Yes I believe the Web account variable should be removed as there seems to be no strong 
#correlation between having a Web account and if a customer has a store credit card

X = pd.DataFrame(sales_train['Days'])
X= sm.add_constant(X)
y = pd.DataFrame(sales_train['CC'])

logreg02 = sm.Logit(y, X).fit()
logreg02.summary2()

#15. Write the descriptive form of the logistic regression using the coefficients
#obtained from Question 1

# CC = B0 + B1(Days)

#16. Validate the model using the test data set

X_test = pd.DataFrame(sales_test['Days'])
X_test = sm.add_constant(X_test)
y_test = pd.DataFrame(sales_test['CC'])

logreg02_test = sm.Logit(y_test, X_test).fit()
logreg02_test.summary2()

#17. Obtain the predicted values of the response variable for each record in the data set. 
y_pred = logreg02.predict(X_test)
y_pred = (y_pred >= 0.5).astype(int)
accuracy = (y_pred == y_test['CC']).mean()
accuracy

