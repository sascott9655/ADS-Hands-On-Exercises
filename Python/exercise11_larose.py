# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:12:22 2026

@author: samsc
"""

#34, 35, 36, 37, 38, 39, 40, & 41 

import pandas as pd
import numpy as np
import statsmodels.api as sm


bank_reg_tr = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/bank_reg_training.csv')
bank_reg_tr
bank_reg_tst = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/bank_reg_test.csv')
bank_reg_tst
#34. Use the training set to run a regression predicting Credit Score, based on 
#Debt-to-Income Ratio and Request Amount. Obtain a summary of the model. Do both predictors
#belong in the model?

X = pd.DataFrame(bank_reg_tr[['Debt-to-Income Ratio', 'Request Amount']])
y = pd.DataFrame(bank_reg_tr[['Credit Score']])

X = sm.add_constant(X)

model34 = sm.OLS(y, X).fit()

model34.summary()

#I believe that the Request Amount predictor does not belong in the model. When I removed it
#from the model the risk of multicollinearity went down. I also think logically that Request Amount has
#less to do with Credit score than debt-to-income ratio. So I think only the debt-to-income ratio
#belongs.

#35. Validate the model from the previous exercise. 

X_test = pd.DataFrame(bank_reg_tst[['Debt-to-Income Ratio', 'Request Amount']])
y_test = pd.DataFrame(bank_reg_tst[['Credit Score']])
X_test = sm.add_constant(X_test)
model34_test = sm.OLS(y_test, X_test).fit()
model34_test.summary()                                

#36.Use the regression score to complete this sentence: "The estimated Credit Score equals..."

#The estimated Credit Score equals 0.038 R squared. 

#37. Interpret the coefficient for Debt-to-Income Ratio.

#The Debt-to-Income ratio have a negative correlation with Credit Score which is why
#the coefficient is represented by a negative number. The more debt the lower the Credit score
#the lower debt the higher the credit score.

#38. Interpret the coefficient for Request Amount

#When the debt goes up the credit score goes down and therefore the
#request amount is lowered. However when debt is lowered the request amount
#goes up keeping it at a balance near 0.

#39. Find and interpret the value of s.
#Std error is how much the predictor can accurately predicts the Credit Score. In the
#model the std error the const has 1.328 error and Debt-to-Income Ratio has an error
#value of 4.826. However the Request Amount has an error score that is very small at 6.85e-05. 
#This means that the Credit Score is most impacted by error scores that are higher. 

#40. Find and interpret R squared.
#R squared is how accurate the model is with all of its dependent variables in consideration.
#It considers how well the model predicts the independent variable. An R squared value of 0.038 is
#really low meaning that the model needs help or more information to help predict accurately the
#target variable.

#41. Find MAE baseline and MAE regression and determine whether the regression model outperformed
#its baseline model. 

#MAE baseline
(y - y.mean()/len(bank_reg_tst)).mean() #673.99 for Credit Score close to const which is 665
#MAE regression
y_hat = model34.predict(X)
print((y_hat.mean())) #674.06
#MAE regression slightly outperforms MAE baseline.



