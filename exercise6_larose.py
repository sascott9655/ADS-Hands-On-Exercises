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

adult_tr = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/adult.csv')
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
mar_cat_pd = pd.get_dummies(adult_tr['martial-status'], drop_first=True).astype(int)

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

