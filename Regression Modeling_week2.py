# -*- coding: utf-8 -*-
"""
# Course: Regression Modeling in Practice
# Week2
# Editor: Kuo-Lin Hsueh 
"""
import numpy as np
import pandas as pandas
import seaborn
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels.formula.api as smf

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#call in data set
data = pandas.read_csv('ool_pds.csv')
data= data.dropna()
# convert variables to numeric format using convert_objects function
data['W1_A1'] = data['W1_A1'].convert_objects(convert_numeric=True)
data['W1_A10'] = data['W1_A10'].convert_objects(convert_numeric=True)

#rename the variables
data = data.rename(columns={'W1_A1':'po_interest','W1_A10':'po_discussion'})

#recode the categorical variable

def interest2groups(row):
    if row['po_interest'] <= 4:
        return "1"
    elif row['po_interest'] == -1:
        return np.nan
    else:
        return "0"
        
data['interest2groups'] = data.apply(lambda row : interest2groups(row),axis=1)
data['interest2groups'] = data['interest2groups'].convert_objects(convert_numeric=True)
data= data.dropna()

############################################################################################
# BASIC LINEAR REGRESSION
############################################################################################
scat1 = seaborn.barplot(x="interest2groups", y="po_discussion", data=data)
plt.xlabel('Politics Interest')
plt.ylabel('Politics Discussion Frequency')
#plt.title (' ')
print(scat1)

print ("OLS regression model for the association between politics interest and politics discussion")
reg1 = smf.ols('po_discussion ~ interest2groups', data=data).fit()
print (reg1.summary())



