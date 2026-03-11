# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:17:57 2026

@author: samsc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tools.tools as stattools
from sklearn.tree import DecisionTreeClassifier, export_graphviz

adult_tr = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/adult.csv')
adult_tr

#14. Create a CART model using the training data set that predicts income using marital status
#and capital gains and losses. Visualize the decision tree (that is, provide the decision tree
#output). Describe the first few splits in the decision tree.

#For simplicity: 
y = adult_tr['income']

adult_tr['Cap_Gains_Losses'] = adult_tr['capital-gain'] + adult_tr['capital-loss']
adult_tr

#We have categorical variable Marital status, among our predictors. The CART model
#implementedin the sklearn package needs categorical variables converted to a dummy
#variable form. Thus, we will make a series of dummy variables for Marital Status 
#using categorical() command(Larose, 2019).

#Making an array to make a matrix of dummy variables and dictionary

#I would get error called: NotImplementedError: categorical has been removed. This was using
#Larose's technique in the book. However I was recommend by Microsoft Copilot to use more modern
#function called pd.getDummies(). So I switched it up. 
mar_cat_pd = pd.get_dummies(adult_tr['marital-status'], drop_first=True).astype(int)

#Larose(2019 code:
#mar_np = np.array(adult_tr['martial-status'])
#(mar_cat, mar_cat_dict) = stattools.categorical(mar_np, drop=True, dictnames=True)

#They are 4 columns in mar_cat matrix because they are 5 values in marital-status. Each value
#has its own column and is represented differently as a vector. Mar_cat_dict tells us
#which dummy variable is which.

#adding dummy variables into a new dataframe
X = pd.concat((adult_tr[['Cap_Gains_Losses']], mar_cat_pd), axis=1)
X
#Larose(2019) code:
#mar_cat_pd = pd.DataFrame(mar_cat)
#X=pd.concat((adult_tr[['Cap_Gain_Losses']], mar_cat_pd), axis=1)


#Clarify column names in X so that is it easier to read tree
X_names = ["Cap-Gains-Losses", "Married-ArmedForces", "Married", "Divorced", "Never-married", "Separated", "Widowed"]
y_names = ['<=50K', ">50K"]

cart14 = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=5).fit(X, y)

#save tree structure and show it:
    
export_graphviz(cart14, out_file = "C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/cart14.dot", feature_names=X_names,
                class_names=y_names)

#Copilot told me to run this command: dot -Tpng "C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/cart14.dot"
# -o "C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/cart14.png" to visualize tree using graph_viz
#Will attach cart.png to hw assignment for clarity

predIncomeCART = cart14.predict(X)

#First decision split: The CART algorithm splits on if the adult is married
#on a 0.5 probability. The class predicts that based on the marriage split question that 
#people will be categorize as income with <=50K. If True, the algorithm tests on if the capital-gain-loss
#is less than or 7073.5. If False, it will test capital is less than or equal to
#5095.5. Both these splits show that it will stil classify adults who are under <=50K 
#of income. The algorithm will stop splitting until it finds at least one leaf node
#that will have an income of >50K. 

#15. Develop a CART model using the test data set that utilizes the same target and predictor
#variables. Visualize the decision tree. Compare the decision trees. Does the test data result
#match the training data result?


adult_tst = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/Data_Sets/adult_ch6_test.csv')
adult_tst


#Same process as step 14.

y2 = adult_tst['income']

adult_tst['Cap_Gains_Losses'] = adult_tst['capital-gain'] + adult_tst['capital-loss']
adult_tst

mar_cat_pd2 = pd.get_dummies(adult_tst['marital-status'], drop_first=True).astype(int)

X2 = pd.concat((adult_tst[['Cap_Gains_Losses']], mar_cat_pd2), axis=1)
X2


cart15 = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=5).fit(X2, y2)

export_graphviz(cart15, out_file = "C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/cart15.dot", feature_names=X_names,
                class_names=y_names)

predIncomeCART = cart15.predict(X2)

#The test results very much match the training data results. I assume the model is accurate. The only difference
#really being that the numbers will be slightly different due to the test set having about half as many instances.

#16. Use the training data set to build a C5.0 model to predict income using marital
#status and capital gains and losses. Specify the minimum of 75 cases per terminal mode.
#Visualize the decision tree. Describe the first few splits in the decision tree.

c50_16 = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=5, min_samples_leaf=75).fit(X, y)

export_graphviz(c50_16, out_file = "C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/c50_16.dot", feature_names=X_names,
                class_names=y_names)

c50_16.predict(X)

# The C5.0 model the same model splits as the CART model. It splits on married <= 0.5. 
# It then classifies that the model will predict income is <=50K a year whether True or
# False. The Cap-Gains_Losses variable is married <= 0.5 is true has the split question
# of is it less than Cap-Gains-Losses of 7073.5. If married <=0.5is False then the split
# turns into is Cap-Gains_losses <= 5095.5? These two splits still then classify the income
# of the adult as less than <=50K. The C5.0 algorithm keeps going until each reaches a classification
# of > 50K. 

#17. How does your C5.0 model compare to the CART model? Describe the similarities and differences.

#Both the CART model and the C5.0 model are very similar. They both have the same test splits with
#results appearing to be very similar. The use of gini vs entropy for measurement loss seems like they are
#conversions of one another. The sampling distributions are also the same. The only real difference is that one
#uses entropy and the other uses gini.



