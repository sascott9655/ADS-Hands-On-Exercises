# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 06:17:20 2026

@author: samsc
"""

import numpy as np
import pandas as pd

bank = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/ADS-Hands-On-Exercises/bank.csv', sep = ';')

bank.info()


#create crosstab with how many times clients were contacted before bank campaign and the number of yes's
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