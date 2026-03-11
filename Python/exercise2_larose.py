# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:17:20 2026

@author: samsc
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

bank = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/bank.csv', sep = ';')
bank

bank.info()


#create contingency table with how many times clients were contacted before bank campaign and the number of yes's
crosstab01 = pd.crosstab(bank['previous'], bank['y'])
crosstab01

#Create a dataframe witht the first nine records
first_nine = bank[:9]
first_nine

#Create a dataframe with age and marital records only
age_marital = bank[['age', 'marital']]
age_marital

#Save the first 3 records of the age_marital in its own dataframe
first_three_am=age_marital[:3]
first_three_am

#needed package openpyxl to read xlsx files
adult = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/adult.csv')

adult.info()

table01 = pd.crosstab(adult['sex'], adult['workclass'])
table01

table02 = pd.crosstab(adult['sex'], adult['martial-status'])
table02

#Display sex and workclass values of the person in the first record. What cell 
#do they belong to? How many other records in the data set have the same
#combinations of sex and workclass values?

first_adult=adult.iloc[0]
sex=first_adult['sex']
workclass=first_adult['workclass']
first_adult

count = table01.loc[sex, workclass]
count

print(f"The first person is {sex} and works in {workclass}.")

#The cell they belong to is the male state-gov cell. 809 total records are males who work in State-gov

#Display the sex and marital status values of the people in records 6-10. Which
#cells of table02 do they belong to? How many other records in the data set have 
#the same combinations of sex and marital status values?


#MY ATTEMPT-----------------------------------------------------------------
#six_thru_ten = adult[6:11]
#six_thru_ten
#count2 = table02.loc[six_thru_ten['sex'], six_thru_ten['martial-status']]
#count2
#----------------------------------------------------------------------------


#COPILOTS assistance
six_thru_ten = adult.iloc[6:11] 
six_thru_ten

results = []

for _, row in six_thru_ten.iterrows():
    sex_val = row['sex']
    marital_val = row['martial-status']
    count = table02.loc[sex_val, marital_val]
    results.append((sex_val, marital_val, count))
    
results

#Cells 2 female, 3 male. Female 1 marital status is Married-spouse-absent with 205 records and 
#Female 2 is Never-married with 4767 records. Male 1 is married-civ-spouse,
#Male 2 is married-civ-spouse, Male 3 is married-civ-spouse all with 13319 records.

#Create new data set with only records whose marital status is "Married-civ-spouse"
#and name the data set adultMarried.

adultMarried_mask = adult['martial-status'] == ' Married-civ-spouse'
adultMarried_mask
adultMarried=adult[adultMarried_mask]
adultMarried

#Recreate the contingency table of sex and workclass using the adultMarried data
#set, What differences do you notice between the sexes?

aMcontin_table = pd.crosstab(adultMarried['sex'], adultMarried['workclass'])
aMcontin_table

#More males are employed than females in married civilian spouse relationships

#Create a new data set for adults over the age of 40 called adultOver40.
adult_age_mask = adult['age'] > 40
adultOver40 =adult[adult_age_mask]

#Create contingency table for the adults over 40 years old. What difference
#does it make?
over40contin_table = pd.crosstab(adultOver40['sex'], adultOver40['workclass'])
over40contin_table

#Not too much of a difference.
