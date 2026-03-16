# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 09:47:00 2026

@author: samsc
"""
#28, 29, 30, 31, 32, 33, & 34 

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
#true churn records you need to resample in order to have 20% of the 
#rebalanced data set have true churm values
type(train_churn_true)
tot_true_churn_vals= train_churn_true + test_churn_true
p = 0.20
tot_records = len(churn)

churn_records_needed = (p *tot_records - tot_true_churn_vals) / (1 - p)
churn_records_needed #about 230 records

#31. Perform the rebalancing described in the previous exercise and confirm 
# that 20% of the rebalanced data set have true churn values

churn['Churn'].value_counts()

ratio = churn['Churn'].value_counts()[1] / churn.shape[0] * 100
ratio #about 14 percent have churn values that are true

to_resample = churn.loc[churn['Churn'] == True]

our_resample = to_resample.sample(n=230, replace=True)

churn_rebal = pd.concat([churn, our_resample], axis=0)

churn_rebal['Churn'].value_counts()

ratio = churn_rebal['Churn'].value_counts()[1] / churn_rebal.shape[0] * 100
ratio #20 percent

#32. Which baseline model do we use to compare our classification model 
# performance against? To which value does this baseline model assign all 
# predictions? What is the accuracy of this baseline model?

#We use a binary classifier model and get the majority class and assign the
#accuracy of predicting the majority class as our baseline model.

#The value would be when Churn is equal to False
 
baseline_acc = churn_rebal['Churn'].value_counts()[0]/ churn_rebal.shape[0] * 100
baseline_acc.round(1)
#the accuracy of this baseline model is 80 percent

#33. Validate your partition by testing in mean day minutes for the training set
#versus the test set?

churn_tr['Day Mins'].mean() #179
churn_tst['Day Mins'].mean() #179

#both 179  minutes meaning the train_test_split is not too biased

#34. Validate your partition by testing for the difference in proportion of true
#churn records for the training set versus the test set.

churn_tr['Churn'].mean() 
#14%
churn_tst['Churn'].mean() #14%

