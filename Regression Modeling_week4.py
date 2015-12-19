# -*- coding: utf-8 -*-
"""
# Course: Regression Modeling in Practice
# Week4
@author: Kuo-Lin
"""

import numpy
import pandas
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf 

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)


##############################################################################
# DATA MANAGEMENT
##############################################################################

#call in data set
data = pandas.read_csv('ool_pds.csv')
data= data.dropna()
# convert variables to numeric format using convert_objects function
data['W1_A1'] = data['W1_A1'].convert_objects(convert_numeric=True)
data['W1_A10'] = data['W1_A10'].convert_objects(convert_numeric=True)

#rename the variables
data = data.rename(columns={'W1_A1':'po_interest','W1_A10':'po_discussion',
                            'PPETHM':'race', 'PPEDUC':'education', 
                            'PPGENDER':'gender', 'PPAGECT4':'age4',
                            'PPINCIMP':'income'})




##############################################################################
# END DATA MANAGEMENT
##############################################################################

##############################################################################
# CATEGORICAL VARIABLES WITH 3+ CATEGORIES
##############################################################################
#recode the categorical variable
# here I simply recode 1 is intersted; 0 is not interested. 
def interest2groups(row): # 1 is extremely interested ; 5 is not at all...
    if row['po_interest'] <= 4:
        return "1"
    elif row['po_interest'] == -1:
        return np.nan
    else:
        return "0"

data['interest2groups'] = data.apply(lambda row : interest2groups(row),axis=1)
data['interest2groups'] = data['interest2groups'].convert_objects(convert_numeric=True)
data['interest2groups']= data['interest2groups'].dropna()

data['education_c'] = (data['education']-data['education'].mean())

def gender_c(row):
    if row['gender'] == 1:
        return '0'
    elif row['gender'] == 2:
        return '1'

data['gender_c'] = data.apply(lambda row : gender_c(row), axis = 1)
data['gender_c'] = data['gender_c'].convert_objects(convert_numeric=True)
data['gender_c']= data['gender_c'].dropna()


reg1 = smf.ols('po_discussion ~ interest2groups', data=data).fit()
print (reg1.summary())


##############################################################################
# LOGISTIC REGRESSION
##############################################################################

# logistic regression 
lreg1 = smf.logit('interest2groups ~ education_c', data=data).fit()
print (lreg1.summary())

# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg1.params))

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))

# logistic regression 
lreg2 = smf.logit('interest2groups ~ income ', data = data).fit()
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))

lreg3 = smf.logit(formula = 'interest2groups ~  education_c + income', data = data).fit()
print (lreg3.summary())

params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))


lreg4 = smf.logit(formula = 'interest2groups ~  age4', data = data).fit()
print (lreg4.summary())

params = lreg4.params
conf = lreg4.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))
