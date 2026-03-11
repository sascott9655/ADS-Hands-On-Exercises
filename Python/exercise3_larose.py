# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:06:56 2026

@author: samsc
"""

import numpy as np
import pandas as pd
from scipy import stats

bank = pd.read_csv('C:/Users/samsc/Desktop/ADS-502-Hands-On-Exercises/bank.csv', sep = ';')
bank

#11. Derive an index field and add it to the dataset
print(bank.size) #Size is bank.shape rows * columns
print(bank.shape)
bank['index'] = range(0, bank.shape[0])
bank

#12. For the days_since_previous field, change the field value 999 to the appropriate code for
#missing values. (This dataset does not have this problem so it will be commented out!)

####bank['days_since_previous'] = bank['days_since_previous'].replace({999: np.nan})

#13. For the education field, reexpress the field values as the numeric values shown in Table 3.1
print(bank['education'].unique()) #find out how many unique values are in education field

education_num = {"unknown" : 0,
                 "primary" : 1,
                 "secondary" : 2,
                 "tertiary" : 3
                 }

bank['education'] = bank['education'].replace(education_num)
bank['education']

#14. Standardize the field age. Print out a list of the first 10 records, including the 
#variables age and age_z.

bank['age_z'] =  stats.zscore(bank['age'])
bank['age_z']

print(bank['age'].mean().round(1)) #sanity check

#15.Obtain a listing of all records that are outliers according to the outliers
#according to the field age_z. Print out a listing of the 10 largest age_z values.

outliers1 = (bank['age_z'] > 3)
outliers2 = (bank['age_z'] < -3)
outliers=bank[outliers1 + outliers2]
outliers 

print(outliers.sort_values(['age_z'], ascending=False)[:10])

#There are no outliers lower than -2.09 roughly so we just have positive std deviation outliers.

#16. For the job field, combine the jobs with less than 5% of the records into a field called other.
job_counts = bank['job'].value_counts()
job_counts_percentage =job_counts / job_counts.sum()
job_counts_percentage =job_counts_percentage.round(2)
job_counts_percentage

job_counts_percentage_mask = job_counts_percentage[job_counts_percentage < 0.05].index #Copilot helped me with this line
job_counts_percentage_mask
bank['other'] = bank['job'].replace(job_counts_percentage_mask, 'other')
print(bank['other'].value_counts())

#18. For the variable month, change the field to 1-12, but keep the variable as categorical.
print(bank['month'].unique())

month_num = {"jan": 1,
             "feb": 2,
             "mar": 3,
             "apr": 4,
             "may": 5,
             "jun": 6,
             "jul": 7,
             "aug": 8,
             "sep": 9,
             "oct": 10,
             "nov": 11,
             "dec": 12
             }
bank['month'] = bank['month'].replace(month_num)
bank['month'] = bank['month'].astype('category')
print(bank['month'].dtype)

#19. Do the following for the duration field
    #a. Standardize the field
    #b. Identify how many outliers there are and identify the most extreme outlier.
    
bank['duration_z'] = stats.zscore(bank['duration'])
outliers3 = bank['duration_z'] < -3
outliers4 = bank['duration_z'] > 3
tot_outliers = bank[outliers3 + outliers4]
print(tot_outliers.shape[0]) # how many outliers
most_extreme_outlier = tot_outliers['duration_z'].sort_values(ascending=False).iloc[0]
most_extreme_outlier



