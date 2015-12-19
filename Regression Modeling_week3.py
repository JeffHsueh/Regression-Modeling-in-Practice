# -*- coding: utf-8 -*-

import numpy
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn

# bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%.2f'%x)

data = pandas.read_csv('gapminder.csv')

# convert to numeric format
data['relectricperperson'] = data['relectricperperson'].convert_objects(convert_numeric=True)
data['urbanrate'] = data['urbanrate'].convert_objects(convert_numeric=True)
data['employrate'] = data['employrate'].convert_objects(convert_numeric=True)

# listwise deletion of missing values
sub1 = data[['urbanrate', 'employrate', 'relectricperperson']].dropna()

####################################################################################
# POLYNOMIAL REGRESSION
####################################################################################

# first order (linear) scatterplot
scat1 = seaborn.regplot(x="urbanrate", y="employrate", scatter=True, data=sub1)
plt.xlabel('Urbanization Rate')
plt.ylabel('Employment Rate')

# fit second order polynomial
# run the 2 scatterplots together to get both linear and second order fit lines
scat1 = seaborn.regplot(x="urbanrate", y="employrate", scatter=True, order=2, data=sub1)
plt.xlabel('Urbanization Rate')
plt.ylabel('Employment Rate')

# center quantitative IVs for regression analysis
sub1['urbanrate_c'] = (sub1['urbanrate'] - sub1['urbanrate'].mean())
sub1['relectricperperson_c'] = (sub1['relectricperperson'] - sub1['relectricperperson'].mean())
sub1[["urbanrate_c", "relectricperperson_c"]].describe()

# linear regression analysis
reg1 = smf.ols('employrate ~ urbanrate_c', data=sub1).fit()
print (reg1.summary())

# quadratic (polynomial) regression analysis

# run following line of code if you get PatsyError 'ImaginaryUnit' object is not callable
#del I
#reg2 = smf.ols('femaleemployrate ~ urbanrate_c + I(urbanrate_c**2)', data=sub1).fit()
#print (reg2.summary())

####################################################################################
# EVALUATING MODEL FIT
####################################################################################

# adding internet use rate
reg3 = smf.ols('employrate  ~ urbanrate_c + I(urbanrate_c**2) + relectricperperson_c', 
               data=sub1).fit()
print (reg3.summary())

#Q-Q plot for normality
fig4=sm.qqplot(reg3.resid, line='r')
print (fig4)
# simple plot of residuals
stdres=pandas.DataFrame(reg3.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')


# leverage plot
fig3=sm.graphics.influence_plot(reg3, size=8)
print(fig3)
