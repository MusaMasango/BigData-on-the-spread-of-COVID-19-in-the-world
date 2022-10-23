
## BigData on the spread of COVID-19 in the world

## Abstract
This project is dedicated to conducting a basic analysis of BigData on the spread of COVID-19 in the world. We will learn how to make predictions based on linear regression, obtain statistics and create interactive maps showing the dynamics of the virus spread.

## Introduction
Nowadays, there is a lot of open data about the spread of COVID-19 in the world. However, few tools are presented to predict and visualize these processes.
This project will show how you can download data from open sources, perform preliminary data analysis, transform and clear data, perform correlation and lag analysis.

Next, we will consider 2 different mathematical approaches to the calculation of a forecast based on linear regression.

To do this, the division of the DataSet into training and test sets will be demonstrated. It will be shown how to build models using 2 different frameworks. Then we will build a forecast and analyze the accuracy and adequacy of the obtained models.

At the end of the project, we will visualize the dynamics of COVID-19 infection spread on interactive maps.

## Code and Resources used

**Python Version**:3.9.12 

**Packages**:pandas,numpy,sklearn,matplotlib,seaborn

**Data Source**:https://ourworldindata.org/coronavirus 

## Data Collection
The data used in this project was downloaded from  [https://ourworldindata.org/coronavirus](https://ourworldindata.org/coronavirus?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01). I then read the csv file using the pd.read_csv() command.

## Data Cleaning
After downloading the data, I needed to clean it up so that it was usable for our model. I made the following changes
* Removed columns with the majority of the NaN values
* Replaced the columns with few missing values I replaced the missing values with either the most occuring entry(mode) for categorical data and with the mean value for numeric data. 
* Changed the data types of columns into the correct ones (i.e object for categorical data and float/int for numeric data)

## Exploratory Data Analysis (EDA)
I looked at the relationship between the different continents and the total cases. Below are highlights from the pivot table

![covid cases](https://github.com/MusaMasango/BigData-on-the-spread-of-COVID-19-in-the-world/blob/main/pivot%20table.png)
![pivot table](https://github.com/MusaMasango/BigData-on-the-spread-of-COVID-19-in-the-world/blob/main/covid%20cases.png)

## Model Building 
The first step of the model building was hypthothesis creation. There are two methods that I used to test my hyphothesis, namely
* Creating models using sklearn
* Time series

First I formulated an hyphothesis based on the number of cases in Africa and the other continents. I then split the data into train and test sets with a test size of 30%. I used the linear regression model then evaluated it using the Mean Absolute Error, Mean Squared Error and Root Mean Squared Error. I then compared the linear regression model with the statsmodel obtained from the statsmodel.api framework. The predicted values from these models are different from the actual values with some uncertainty.

Secondly I used the time series method to test my hyphothesis. In this case we only consider one time series since we are dealing with Africa. I then evaluated it using the Mean Absolute Error, Mean Squared Error and Root Mean Squared Error. The predicted values obtained using the time series are closer to the actual values. 

## Model Performance
Out of the two methods, the time series performed better with an Mean Absolute Error: 765635.4335892488 when compared to the linear regression with an Mean Absolute Error (test): 162489418.69861022

## Interactive maps

During the last part of the project, I produced interactive maps to show the spread of covid-19 on various european countries. The various steps in this process include
*   data transformation for mapping
*   downloading polygons of maps
*   building interactive maps



The statistical data is obtained from [https://ourworldindata.org/coronavirus](https://ourworldindata.org/coronavirus?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01) under the Creative Commons BY license.

