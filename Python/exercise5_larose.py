# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 09:47:00 2026

@author: samsc
"""

#28. Partition the data set, so that 67% of the records are included in training
# data set and 33% are in the test data set. Use a bar graph to confirm your
# proportions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random


churn = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/churn.csv')
churn

#train test split
churn_tr, churn_tst = train_test_split(
    churn, test_size=0.33, random_state=7)

#sanity check to make sure train_test_split works as intended
churn.shape[0]
churn_tr.shape[0]
churn_tst.shape[0]

plt.bar(['Training Set', 'Test Set'], [len(churn_tr), len(churn_tst)])
plt.show()


#29. Identify the total number of records in the training data set and how many
#records in the training set have a churn value of true.

train_churn_true = (churn_tr['Churn'] == True).sum() #320
test_churn_true = (churn_tst['Churn'] == True).sum() #163

#30. Use your answers from the previous exercise to calculate how many
#true churn records you need to resample in order to have 205 of the 
#rebalanced data set have true charm values


