#!/usr/bin/env python
# coding: utf-8

# ## BigData on the spread of COVID-19 in the world
# 

# ## Abstract
# 

# This project is dedicated to conducting a basic analysis of BigData on the spread of COVID-19 in the world. We will learn how to make predictions based on linear regression, obtain statistics and create interactive maps showing the dynamics of the virus spread.
# 

# ## Introduction
# 

# Nowadays, there is a lot of open data about the spread of COVID-19 in the world. However, few tools are presented to predict and visualize these processes.
# This laboratory work will show how you can download data from open sources, perform preliminary data analysis, transform and clear data, perform correlation and lag analysis.
# 
# Next, we will consider 2 different mathematical approaches to the calculation of a forecast based on linear regression.
# 
# To do this, the division of the DataSet into training and test sets will be demonstrated. It will be shown how to build models using 2 different frameworks. Then we will build a forecast and analyze the accuracy and adequacy of the obtained models.
# 
# At the end of the laboratory work, we will visualize the dynamics of COVID-19 infection spread on interactive maps.
# 

# ## Materials and methods
# 

# In this lab, we will learn the basic methods of time series forecasting and their visualization on interactive maps. The laboratory consists of three stages:
# 
# *   Download and preliminary analysis of data
# *   Forecasting
# *   Interactive maps
# 
# The first stage will show you how to download data and pre-prepare it for the analysis:
# 
# *   downloading data
# *   changing the data types of columns
# *   grouping data
# *   DataSet transformation
# *   elimination of missing data
# 
# At the stage of forecasting, we will deal with the methods of building and fitting models, as well as with the automation of statistical information calculation, in particular:
# 
# *   hypothesis creation
# *   splitting the DataSet into training and test sets
# *   building models using 2 different frameworks
# *   calculation of basic statistical indicators
# *   forecasting time series
# 
# At the stage of interactive maps, we will show how to display statistical information on interactive maps:
# 
# *   data transformation for mapping
# *   downloading polygons of maps
# *   building interactive maps
# 

# The statistical data is obtained from [https://ourworldindata.org/coronavirus](https://ourworldindata.org/coronavirus?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01) under the Creative Commons BY license.
# 

# ## Prerequisites
# 
# *   Python,
# *   [Pandas](https://pandas.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01),
# *   [SeaBorn](https://seaborn.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01),
# *   Statistics,
# *   [Plotly](https://plotly.com/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)
# 

# ## Objectives
# 

# After completing this lab, you will be able to:
# 

# *   Download a DataSet from \*.csv files
# *   Automatically change data in the DataSet
# *   Transform a table
# *   Visualize data with pandas and seaborn
# *   Make foreacast models
# *   Build Interactive Maps
# 

# ## Download and preliminary analysis of data
# 

# ### Downloading data
# 

# Some libraries should be imported before you can begin.
# 

# In[1]:


import pandas as pd
import numpy as np


# The next step is to download the data file from the [open repository produced by Our World in Data under the Creative Commons BY license](https://ourworldindata.org/coronavirus?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01) by the **[read_csv()](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**
# 

# In[2]:


covid_word = pd.read_csv("owid-covid-data.csv")
covid_word.head()


# In[3]:


# evaluating missing data
missing_data = covid_word.isnull()
missing_data.head(5)


# In[4]:


# counting the missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# Let's study the DataSet. As you can see, the DataSet consist of 86202 rows × 59 columns. The first 3 columns contain Geo information. Column 4 - date of measurement. Another 55 - COVID-19 data. Also some missing data are observed in the DataSet. We should make sure that Python recognized the types of data correctly. To do this, we should use **[pandas.info()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01&highlight=info#pandas.DataFrame.info)**.
# 

# In[5]:


covid_word.info()


# In[6]:


#drop the entire column with Nan for the following columns
covid_word.drop(['weekly_icu_admissions'], axis=1, inplace=True)
covid_word.drop(['weekly_icu_admissions_per_million'], axis=1, inplace=True)
covid_word.info()


# In[10]:


covid_word['tests_units'].value_counts() # number of entries per category 
covid_word['continent'].value_counts() # number of entries per category
#replace the Nan values with the occuring entry
covid_word['tests_units'].replace(np.nan, "tests performed", inplace=True)
covid_word['continent'].replace(np.nan, "Africa", inplace=True)
covid_word.info()


# In[11]:


# drop the entire columns with Nan
covid_word.drop(['excess_mortality_cumulative_absolute'], axis=1, inplace=True)
covid_word.drop(['excess_mortality_cumulative'], axis=1, inplace=True)
covid_word.drop(['excess_mortality'], axis=1, inplace=True)
covid_word.drop(['excess_mortality_cumulative_per_million'], axis=1, inplace=True)
covid_word.info()


# In[15]:


# replace the Nan value with the mean for the following columns
var = covid_word.loc[:, 'total_cases':'tests_per_case']
temp = covid_word.loc[:, 'total_vaccinations':]
pd.options.display.float_format = '{:,.0f}'.format
avg_var = var.mean(axis=0) # mean value for each column
avg_temp = temp.mean(axis=0) # mean value for each column 
var.replace(0, avg_var, inplace=True)
var.replace(np.nan, avg_var, inplace=True)
temp.replace(0, avg_temp, inplace=True)
temp.replace(np.nan, avg_temp, inplace=True)
covid_word.loc[:, 'total_cases':'tests_per_case'] = var
covid_word.loc[:, 'total_vaccinations':] = temp
covid_word.info()


# In[18]:


covid_word.dtypes


# As you can see, 54 columns of COVID-19 data were recognized correctly (float64). First 4 columns and tests_units were recognized as objects. Let's investigate them:
# 

# In[24]:


covid_word[['iso_code', 'continent', 'location', 'tests_units']]


# In[12]:


covid_word['date']


# ### Сhanging the data types of columns
# 

# As you can see, the columns 'iso_code', 'continent', 'location', 'tests_units' have many repetitions and should be assigned to categorical fields **([pandas.astype()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01&highlight=astype#pandas.DataFrame.astype))**.
# The 'data' field should be converted into DataTime type **([pandas.to_datetime()](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01))**. To see the results, we can use **[pandas.describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01&highlight=describe#pandas.DataFrame.describe)**.
# 

# In[20]:


covid_word[['iso_code', 'continent', 'location', 'tests_units']] = covid_word[['iso_code', 'continent', 'location', 'tests_units']].astype("category")
covid_word.loc[:, 'date'] = pd.to_datetime(covid_word['date'])
covid_word[['iso_code', 'continent', 'location', 'tests_units']].describe()


# As we can see, the DataSet contains information about 6 continents and 219 countries.
# The 'tests_units' field consist of 4 categories. To show them, we can use **[pandas.Series.cat.categories](https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.categories.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# 

# In[25]:


covid_word['location'].cat.categories


# ### Grouping data
# 

# Let's determine how many records of each category there are in the DataSet **[pandas.Series.value_counts()](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)** and show the results in a table **[pandas.Series.to_frame()](https://pandas.pydata.org/docs/reference/api/pandas.Series.to_frame.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# 

# In[27]:


covid_word['tests_units'].value_counts().to_frame()


# As shown above, the DataSet contains 54 statistical fields that can be used for any analysis.
# For simplicity, choose the most informative one - **total cases**.
# Let's determine how many sick people belong to each of the categories using **[pandas.DataFrame.groupby()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01&highlight=groupby#pandas.DataFrame.groupby)** and view it in descending order with the help of **[pandas.DataFrame.sort_values()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# 

# In[29]:


covid_word.groupby('continent')['total_cases'].sum().sort_values(ascending=False).to_frame() 


# ### DataSet transformation
# 

# Let's try to predict the spread of COVID-19 on different continents. To do this, we need to transform our DataSet. In particular, the dates of measurement should be used in the index field, and the data on the total cases depending on the continent should be takes as columns. Let's use a pyvot table to do it: **[pivot_table()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01&highlight=pivot_table#pandas.DataFrame.pivot_table)**.
# 

# In[31]:


p_covid = pd.pivot_table(covid_word, values= 'total_cases', index= ['date'], columns=['continent'], aggfunc='sum', margins=False)
p_covid


# We created a new DataSet that will be used for forecasting. Let's visualize this data by **[pandas.DataFrame.plot()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01&highlight=plot#pandas.DataFrame.plot)**. It should be noted, that pandas incapsulate matplotlib library and inherit the function plot(). Therefore, to show this plot, we need to import matplotlib library and apply function **[matplotlib.pyplot.show()](https://matplotlib.org/stable/api/\_as_gen/matplotlib.pyplot.show.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# 

# In[32]:


p_covid.plot()
import matplotlib.pyplot as plt
plt.title('Total cases per continent')
plt.xlabel('date')
plt.ylabel('sum of total cases')
plt.show()


# ## Forecasting
# 

# ### Hypothesis creation
# 

# Before making a forecast, you should first determine the target (output) field for which the forecast will be built. The next step is to create a hypothesis that involves determining the input fields on which our target depends. Let's try to make a prediction about total cases in Africa. We can propose two hypotheses:
# 
# 1.  The number of total cases in Africa depends on the one on other continents.
# 2.  The number of total cases in Africa doesn't depend on the one on other continents.
# 
# To check the first hypothesis, we should make a correlation analysis using **[pandas.DataFrame.corr()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01&highlight=corr#pandas-dataframe-corr)**.
# 

# In[33]:


pd.options.display.float_format = '{:,.2f}'.format
p_covid.corr()


# Each cell contains the correlation coefficients between two columns. Therefore, diagonal elements are equal to one. As can be seen from the Africa column (or row), all the correlation coefficients are more than 0.9. This may be a confirmation of the first hypothesis. Correlation coefficients close to 1 mean the presence of a close linear relationship between the fields. Therefore, to test the first hypothesis, it is convenient to use linear models.
# 

# ### Splitting the DataSet into training and test sets
# 

# In[34]:


proportion_train_test = 0.7
l = int(proportion_train_test * len(p_covid)) # number of rows that will be used as the training 
print("the number of rows that will be used for training is = ", l)


# In[36]:


# Slices:
col = p_covid.columns
X_train, X_test, y_train, y_test = p_covid[col[1:]][:l], p_covid[col[1:]][l:], p_covid[col[0]][:l], p_covid[col[0]][l:] 


# In[37]:


# sklearn function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(p_covid[col[1:]], p_covid[col[0]], test_size=0.3, shuffle=False)
print("number of test samples :", X_test.shape[0])
print("number of training samples:",X_train.shape[0])


# ### Creating models using sklearn
# 

# To build a linear model, it is necessary to create the linear model itself, fit it, test it, and make a prediction.
# To do this, use **[sklearn.linear_model.LinearRegression()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# 

# In[38]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)


# ### Calculation of basic statistical indicators
# 

# The prediction results for the training and test sets are in **y_pred_test** and **y_pred_train** variables. After that, we can check adequacy and accuracy of our model using **[sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**. Also we can get the parameters of the linear model.
# 

# In[39]:


from sklearn import metrics
print("Correlation train", regressor.score(X_train, y_train))
print("Correlation test", regressor.score(X_test, y_test))
print("Coefficients:", regressor.coef_)
print("Intercept", regressor.intercept_)
# pair the feature names with the coefficients
print('Pair the feature names with the coefficients:')
for s in zip(col[1:], regressor.coef_):
    print(s[0], ":", s[1])
print('Mean Absolute Error (train):', metrics.mean_absolute_error(y_train, y_pred_train))
print('Mean Absolute Error (test):', metrics.mean_absolute_error(y_test, y_pred_test))
print('Mean Squared Error (train):', metrics.mean_squared_error(y_train, y_pred_train))
print('Mean Squared Error (test):', metrics.mean_squared_error(y_test, y_pred_test))
print('Root Mean Squared Error (train):', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
print('Root Mean Squared Error (test):', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))


# ### Creating models using statsmodels
# 

# As you can see, there is a big difference in accuracy between the training and test results. It means that this hypothesis is not correct.
# Besides, this framework cannot generate a summary report.
# To do this, we can use the **[statsmodels.api](https://www.statsmodels.org/stable/index.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)** framework.
# 

# In[40]:


import statsmodels.api as sm
model = sm.OLS(y_train, X_train)
results = model.fit()
y_pred_test_OLS = results.predict(X_test)
y_pred_train_OLS = results.predict(X_train)
print(results.summary())


# As you can see, this framework uses the same principles for creating and fitting models. It allows us to build a summary report, also you can get all the other stats coefficients in the same way:
# 

# In[41]:


print('coefficient of determination:', results.rsquared)
print('adjusted coefficient of determination:', results.rsquared_adj)
print('regression coefficients:', results.params, sep = '\n')


# We should join the results to compare these two framework models using **[pandas.DataFrame.join()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**:
# 

# In[42]:


df_test = pd.DataFrame({'Actual_test': y_test, 'Predicted_test': y_pred_test, 'Predicted_test_OLS': y_pred_test_OLS})
df_train = pd.DataFrame({'Actual_train': y_train, 'Predicted_train': y_pred_train, 'Predicted_train_OLS': y_pred_train_OLS}) 
df = df_train.join(df_test, how='outer')
df


# As you can see, pandas joins and orders data correctly according to the index field automatically. Therefore, it is very important to check the index field datatype, especially when we deal with datatime.
# 

# Let's visualize the data.
# 

# In[44]:


df.plot(figsize=(20,10))
plt.show()


# You can see that the results of these two models are the same. Also you can see that the forecast on the test data is not perfect.
# To see the difference between our forecast and the real data, we can use **[seaborn.pairplot()](https://seaborn.pydata.org/generated/seaborn.pairplot.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# 

# In[45]:


import seaborn as sns
sns.pairplot(df_test, x_vars=['Actual_test'], y_vars='Predicted_test',  kind='reg', height = 8)
plt.show()


# The real data values are plotted on the horizontal axis and the predicted ones are plotted on the vertical axis.
# The closer the result points are to the diagonal, the better the model forecast is.
# This plot proves our conclusion about the bad forecast quality under this hypothesis.\
# Moreover, in order to make a forecast for the future, you have to know future data for other continents.
# 

# ### Forecasting time series
# 

# Let's try to test the second hypothesis.
# According to it, we need to consider only one time series. In our case - Africa. The only assumption that can be made - the data for today depends on the previous days values. To check for dependencies, it is necessary to analyze correlations between them. This requires:
# 
# 1.  Duplicating the time series of data and moving it vertically down for a certain number of days (lag)
# 2.  Deleting the missing data at the beginning and end (they are formed by vertical shift (**\[pandas.DataFrame.shift()])([https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01))**
# 3.  Calculating the correlation coefficient between the obtained series.
# 
# Since this operation should be performed for different values of the lag, it is convenient to create a separate function:
# 

# In[46]:


def lag_correlation_ts(y, x, lag):
    """
    Lag correlation for 2 DateSeries
    :param y: fixed
    :param x: shifted
    :param lag: lag for shifting
    :return: DataFrame of lags correlation coefficients
    """
    r = [0] * (lag + 1)
    y = y.copy()
    x = x.copy()
    y.name = "y"
    x.name = "x"

    for i in range(0, lag + 1):
        ds = y.copy().to_frame()
        ds = ds.join(x.shift(i), how='outer')
        r[i] = ds.corr().values[0][1]
    r = pd.DataFrame(r)
    r.index.names = ['Lag']
    r.columns = ['Correlation']
    return r


# In[47]:


y_dataset = p_covid[col[0]]
y_dataset


# Let's test a 30-day lag.
# 

# In[48]:


pd.options.display.float_format = '{:,.4f}'.format
l = pd.DataFrame(lag_correlation_ts(y_dataset, y_dataset, 30))
l


# As you can see, the time series data is highly dependent on the data of the previous period. Even with a 30-day lag, there is a close linear relationship.
# 
# To build a linear model of the type input-target, the target should be the data of the original time series, and the input should be the values for the previous days.
# 
# To automate this process, let's create a universal time series transformation function to a DataSet structure.
# 

# In[49]:


def series_to_supervised(in_data, tar_data, n_in=1, dropnan=True, target_dep=False):
    """
    Transformation into a training sample, taking into account the lag
     : param in_data: Input fields
     : param tar_data: Output field (single)
     : param n_in: Lag shift
     : param dropnan: Do destroy empty lines
     : param target_dep: Whether to take into account the lag of the input field. If taken into account, the input will start with lag 1
     : return: Training sample. The last field is the source
    """

    n_vars = in_data.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # for i in range(n_in, -1, -1):
    if target_dep:
        i_start = 1
    else:
        i_start = 0
    for i in range(i_start, n_in + 1):
        cols.append(in_data.shift(i))
        names += [('%s(t-%d)' % (in_data.columns[j], i)) for j in range(n_vars)]

    if target_dep:
        for i in range(n_in, -1, -1):
            cols.append(tar_data.shift(i))
            names += [('%s(t-%d)' % (tar_data.name, i))]
    else:
        # put it all together
        cols.append(tar_data)
        # print(tar_data.name)
        names.append(tar_data.name)
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


# In[50]:


dataset = series_to_supervised(pd.DataFrame(y_dataset), y_dataset, 30)
dataset


# As you can see, the first and last columns contain the same target data. Therefore, similarly to the previous model, we will form training and test DataSets, fit and compare the results.
# 

# In[51]:


col_2 = dataset.columns
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(dataset[col_2[1:-2]], dataset[col_2[0]], test_size=0.3, shuffle=False) 
regressor2 = LinearRegression()
regressor2.fit(X_train_2, y_train_2)
y_pred_test_2 = regressor2.predict(X_test_2)


# <details><summary>Click <b>here</b> for the solution</summary> 
# train_test_split(dataset[col_2[1:-2]], dataset[col_2[-1]], test_size=0.3, shuffle=False)
# </details>
# 

# In[52]:


print("Correlation train", regressor2.score(X_train_2, y_train_2))
print("Correlation test", regressor2.score(X_test_2, y_test_2))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_2, y_pred_test_2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_2, y_pred_test_2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_2, y_pred_test_2)))


# As you can see, the forecast results of the test data set are much better than ones of the previous model. Let's visualize these 2 results:
# 

# In[54]:


y_pred_test_2 = pd.DataFrame(y_pred_test_2, columns = ['Predicted_test time series'])
y_pred_test_2.index = y_test_2.index
df_2 = pd.DataFrame({'Actual_test': y_test, 'Predicted_test': y_pred_test, })
df_2 = df_2.join(y_pred_test_2, how='outer')
df_2.plot(figsize=(20,10))
plt.show()


# As you can see, the second model has made a perfect forecast.
# 

# ## Interactive maps
# 

# ### Data transformation for mapping
# 

# It is convenient to display the spread of viral infection on a map to visualize it. There are several libraries for this. It is convenient to use the library **[plotly.express](https://plotly.com/python/plotly-express/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)** to display the dynamics of COVID-19.
# 

# In[55]:


import plotly.express as px


# Let's build the dynamics of the spread of COVID-19 (total_cases) for European countries. To do this:
# 
# 1.  Filter the initial DataSet to leave only European countries.
# 2.  Leave only the columns with necessary GEO data ("location", "date", "total_cases") ordered by "location" and "date".
# 

# In[56]:


covid_EU = covid_word[covid_word.continent == "Europe"]
covid_EU = covid_EU[["location", "date", "total_cases"]].sort_values(["location", "date"])
covid_EU


# Before visualization, we should delete NaN data:
# 

# In[61]:


modified_confirmed_EU = covid_EU[np.isnan(covid_EU.total_cases) == False]

modified_confirmed_EU


# We should change the type of total_cases column to int64 to display the map colors correctly.
# 

# In[62]:


c = 'total_cases'
modified_confirmed_EU.loc[:, c] = modified_confirmed_EU[c].astype('int64')
modified_confirmed_EU.info()


# Let's group the data by country.
# For each country, the rows should be ordered by the date of measurement.
# 

# In[63]:


modified_confirmed_EU=modified_confirmed_EU.set_index('date').groupby('location')


# As we can see, the DataSet consists of 20052 rows. It is too much for GEO mapping. We need to reduce their number. For example, we can display not by days, but by months. To do this, you need to use the function **[pandas.DataFrame.resample()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# 

# In[64]:


modified_confirmed_EU = modified_confirmed_EU.resample('M').sum()
print(modified_confirmed_EU)


# To display everything on the map correctly, we need to restore the data in the "location" field and duplicate the "Data" field by converting it to the data type str. This is necessary to display the dates of the interactive map on the scroll bar correctly.
# 

# In[66]:


modified_confirmed_EU.loc[:,'location'] = modified_confirmed_EU.index.get_level_values(0)
modified_confirmed_EU.loc[:,'Date'] = modified_confirmed_EU.index.get_level_values(1).astype('str')

print(modified_confirmed_EU)


# ### Downloading polygons of maps
# 

# The next step is to download the map polygons. They are publicly available: [https://data.opendatasoft.com/explore/dataset/european-union-countries%40public/information/](https://data.opendatasoft.com/explore/dataset/european-union-countries%40public/information/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01).
# Also a DataSet schema is presented on this site.
# You can see that the key "NAME" of this json is connected to the field "location" in our DataSet.
# 

# In[80]:


import json
get_ipython().system('wget european-union-countries.geojson "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/data-science-in-health-care-basic-prognostication-and-geo-visualization/european-union-countries.geojson"')
with open("european-union-countries.geojson", encoding="utf8") as json_file:
    EU_map = json.load(json_file)


# The next step is building an interactive map using **[plotly.express.choropleth()](https://plotly.com/python/mapbox-county-choropleth/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**. We should send as input parameters:
# 
# 1.  Polgons of countries: geojson=EU_map,
# 2.  Fields for comparison of countries in the DataSet: locations='location',
# 3.  The key field in the json file that will be compared with the locations: featureidkey='properties.name',
# 4.  The color of countries: color= 'total_cases',
# 5.  Information for the legend: hover_name= 'location', hover_data= \['location', 'total_cases'],
# 6.  Animation field: animation_frame= 'Date',
# 7.  Color scale: color_continuous_scale=px.colors.diverging.RdYlGn\[::-1]
# 
# **Warning: you have to wait a few minutes.**
# 

# In[81]:


fig = px.choropleth(
    modified_confirmed_EU[::-1],
    geojson=EU_map,
    locations='location',
    featureidkey='properties.name',    
    color= 'total_cases', 
    scope='europe',
    hover_name= 'location',
    hover_data= ['location', 'total_cases'],
    animation_frame= 'Date', 
    color_continuous_scale=px.colors.diverging.RdYlGn[::-1]
)


# Then we should change some map features. For example: showcountries, showcoastline, showland, fitbouns in the function **[plotly.express.update_geos()](https://plotly.com/python/map-configuration/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# Also we can modify our map layout: **[plotly.express.update_layout](https://plotly.com/python/creating-and-updating-figures/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)**.
# 

# In[82]:


fig.update_geos(showcountries=False, showcoastlines=False, showland=False, fitbounds="locations")

fig.update_layout(
    title_text ="COVID-19 Spread EU",
    title_x = 0.5,
    geo= dict(
        showframe= False,
        showcoastlines= False,
        projection_type = 'equirectangular'
    ),
    margin={"r":0,"t":0,"l":0,"b":0}
)


# In[83]:


from IPython.display import HTML
HTML(fig.to_html())


# ## Conclusions
# 

# In this laboratory work, we learned how to build hypotheses for forecasting models. We transformed DataSets for input-output models. We learned how to divide DataSets into training and test sets. It was also shown how to predict time series models using lag transformations. At the end of the laboratory work, we displayed the DataSet on a dynamic interactive map in \* .html format.
# 

# ## Authors
# 

# [Yaroslav Vyklyuk, prof., PhD., DrSc](http://vyklyuk.bukuniver.edu.ua/en/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01)
# 

# Copyright © 2020 IBM Corporation. This notebook and its source code are released under the terms of the [MIT License](https://cognitiveclass.ai/mit-license/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01).
# 
