
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
![pivot table](pivot table.png)
![covid cases](covid cases.png)

## Materials and Methods
In this project, we will learn the basic methods of time series forecasting and their visualization on interactive maps. The project consists of three stages:

*   Download and preliminary analysis of data
*   Forecasting
*   Interactive maps

The first stage will show you how to download data and pre-prepare it for the analysis:

*   downloading data
*   changing the data types of columns
*   grouping data
*   DataSet transformation
*   elimination of missing data

At the stage of forecasting, we will deal with the methods of building and fitting models, as well as with the automation of statistical information calculation, in particular:

*   hypothesis creation
*   splitting the DataSet into training and test sets
*   building models using 2 different frameworks
*   calculation of basic statistical indicators
*   forecasting time series

At the stage of interactive maps, we will show how to display statistical information on interactive maps:

*   data transformation for mapping
*   downloading polygons of maps
*   building interactive maps

The statistical data is obtained from [https://ourworldindata.org/coronavirus](https://ourworldindata.org/coronavirus?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinhealthcarebasicprognosticationandgeovisualization26633115-2022-01-01) under the Creative Commons BY license.

