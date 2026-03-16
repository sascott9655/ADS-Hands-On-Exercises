# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:32:47 2026

@author: samsc
"""

#23, 24, 25, 26, 27, 28, 29, & 30
#Question 27 Correction - page 111: 27. Create a cost matrix, called the 3x cost matrix
#that specifies a false negative is three times as bad as a false positive. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix

#23. Using the training data set, create a 5.0 model (Model 1) to predict a 
#customer's Income using Marital Status and Capital Gains and Losses. Obtain a
#predicted responses.

adult_tr = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/adult_ch6_training.csv')
adult_tr
adult_tst = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/adult_ch6_test.csv')
adult_tst


X = adult_tr[['Marital status', 'Cap_Gains_Losses']]
y = adult_tr['Income']

#Because marital status is categorical, we need dummy vars so model can process marital status information
X = pd.get_dummies(X, drop_first=True)

model1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=7).fit(X, y)
train_pred = model1.predict(X)

#24. Evaluate Model 1 using the test data set. Construct a contingency table
#to compare the actual and predicted values of Income

X_test = adult_tst[['Marital status', 'Cap_Gains_Losses']]
y_test = adult_tst['Income']
X_test = pd.get_dummies(X_test, drop_first=True)
pred = model1.predict(X_test)

cm = confusion_matrix(y_test, pred)
cm



#25. For Model 1, recapitulate Table 7.4 from the text, calculating all of the 
# model evaluation measures shown in the table. Call this table the Model Evaluation
# Table. Leave space for model 2

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

#total actually positive 
TAP = TP + FN

#total actually negative
TAN = FP + TN

TPP = FP + TP

TPN = FN + TN

accuracy = (TN + TP) / (TN + FP + FN + TP)
error_rate = 1 - accuracy
sensitivity = TP / TAP
specificity = TN / TAN
precision = TP / TPP
recall = sensitivity
f1 = 2* (precision * recall) / (precision + recall)
f2 = 5 * (precision * recall) / ((4 * precision) + recall)
f05 = 1.25 * (precision * recall) / ((0.25 * precision) + recall)

#microsoft copilot help me make the data dictionary
data = {
    "Evaluation Measure": [
        "Accuracy",
        "Error rate",
        "Sensitivity (Recall)",
        "Specificity",
        "Precision",
        "F1 Score",
        "F2 Score",
        "F0.5 Score"
    ],
    "Value": [
        accuracy,
        error_rate,
        sensitivity,
        specificity,
        precision,
        f1,
        f2,
        f05
    ]
}

model_evaluation_table = pd.DataFrame(data)
model_evaluation_table

#26. Clearly and completely interpret each of the Model 1 evaluation measure
#from the Model Evaluation Table

#Accuracy is 82%. This means that these marital status and cap gain losses combined are
#able to accurately predict the outcome of income 82 percent of the time correctly. That
#is not too bad considering that it is only two predictors

#Error Rate is 18%. It is the opposite of the accuracy so read that for more details

#Sensitivity or Recall shows 29%. This shows the amount of true positive results out
#of all the results that were actually positive or income >50K . This shows that our model had a 
#difficult time predicting positive values and it would probably be neccessary to add
#more samples of >50K data so the model can predict these instances better.

#Specificity showed 99%. This shows the amount of true negative results out of all the 
#results that were actually negative or income <=50K. This shows that the model performed
#really well at predicting outcomes that had income equal to <=50K. This again shows
#the inbalance of how much more data or instances we had were the income is <=50K.

#Precision is 92%.  Precision calculates the true positive over the total predicted positive. 
#The result of this value is high because most of the positive values that were predicted were 
#true positive which made up 434 out of 468 instances of precision. That means our model was
#good at predicting <=50K instances as only 34 of those instances were predicted incorrectly.

#F1 score is 44 percent. This means that precision and recall have equal weight in the
#F1 score. Because the precision is high percentage and recall is low percentage the fl score means out and since
#they are more recall instances, its weight drags the percent to a low 44 percent. This 
#demonstrates that our model did not perform well at predicting income outcomes that were
#>50K

#F2 score is 34 percent. This means that recall has twice the weight or influence  in the
#score. Because the precision is high percentage and recall is percentage the f2 score would
#not be the ideal measure for our model.

#F05 score is 64 percent. This means that precision has twice the weight or influence in the
#score because recall has the half the weight. Because the precision is high percentage
#and recall is percentage the f05 score would be the most ideal measure out of
#the fscores for our model.

#27. Create a cost matrix, called the 3x cost matrix
#that specifies a false negative is three times as bad as a false positive.

threex_cost_matrix = [[0,1],
                     [3,0]]
threex_cost_matrix
      
#28. Using the training set , build a C5.0 model (Model 2) to predict a customer's
#Income using Marital Status and Capital Gains and Losses, using the 3x cost matrix

#Had help from microsoft copilot lines 159-163: 
    
weights = {0: 1, 1: 3} #labeling weights to class value. Because false negative is three times as bad,
#labeling positive with 3 gives its weighted boost. 
adult_tr['weighted_income'] = adult_tr['Income'].map({'<=50K': 0, '>50K': 1}) #using map function to label class values
adult_tst['weighted_income'] = adult_tst['Income'].map({'<=50K': 0, '>50K': 1}) #using another var attribute because it will overwrite
#income value with 0s and 1s if we do not

#--------------------------

y2 = adult_tr['weighted_income']

model2 = DecisionTreeClassifier(criterion='entropy', max_depth=5,class_weight=weights, random_state=7).fit(X, y2)
#using weights to ensure the 3x matrix cost is incorporated

pred2 = model2.predict(X_test)

 
#29. Evaluate your predictions from Model 2 using the actual response values from 
# the test data set. Add Overall Model Cost and Profit per Customer to the Model
#Evaluation Table. Calculate all measures from the Model Evaluation.

cm2 = confusion_matrix(adult_tst['weighted_income'], pred2)
cm2

TN2 = cm2[0][0]
FP2 = cm2[0][1]
FN2 = cm2[1][0]
TP2 = cm2[1][1]

#total actually positive 
TAP2 = TP2 + FN2

#total actually negative
TAN2 = FP2 + TN2

TPP2 = FP2 + TP2

TPN2 = FN2 + TN2

accuracy2 = (TN2 + TP2) / (TN2 + FP2 + FN2 + TP2)
error_rate2 = 1 - accuracy2
sensitivity2 = TP2 / TAP2
specificity2 = TN2 / TAN2
precision2 = TP2 / TPP2
recall2 = sensitivity2
f1two = 2* (precision2 * recall2) / (precision2 + recall2)
f2two = 5 * (precision2 * recall2) / ((4 * precision2) + recall2)
f05two = 1.25 * (precision2 * recall2) / ((0.25 * precision2) + recall2)

#microsoft copilot help me make the data dictionary
data2 = {
    "Evaluation Measure": [
        "Accuracy",
        "Error rate",
        "Sensitivity (Recall)",
        "Specificity",
        "Precision",
        "F1 Score",
        "F2 Score",
        "F0.5 Score"
    ],
    "Value": [
        accuracy2,
        error_rate2,
        sensitivity2,
        specificity2,
        precision2,
        f1two,
        f2two,
        f05two
    ]
}

model_evaluation_table2 = pd.DataFrame(data2)
model_evaluation_table2

#30. Compare the evaluation measures from Model 1 and Model 2 using the 3x cost
#matrix. Discuss the strengths and weaknesses of each model.

#Model 1 had more spuratic stats, or statistics with higher variance across the board.
#Because of the amount of data instances that resulted in income being <=50K more of these
#having no weighted category hurt the model in some areas while excelling in other areas.
#When it came to predicting <=50k model1 is much better at predicting the majority of the
#data. However it poorly predicts data instances with >50K as the result. Model2 is 
# overall a more all-around model. Where it will predict all the stats with no real
#particular strength in predictive power. However model2 does understand the data
#better as it somewhat "resamples" data points with its weighted power. A pretty 
#interesting observation between the two models.















