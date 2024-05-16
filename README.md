# Discover the world of Quant
LSEG Challenge - ESGEM.FGI Price Prediction

> [!NOTE] 
> For a comprehensive understanding, see our detailed documentation available [here](https://github.com/corfesh/LSEG-Quant-Challenge/blob/main/LSEG%20Challenge%20Report.pdf).

## Challenge Description

We were required to create a predictive model using historical prices to forecast the future performance of an equity index. 

The input data consists of time-series data and the output data consists of index predictions for the next 5 days (approx. 1 trading week)

## FTSE ESG Emerging Index

The financial instrument in scope is FTSE ESG Emerging Index, which is part of the FTSE ESG Index Series.

The series is designed to help investors align investment and ESG considerations into a broad benchmark. The indices provide risk/return characteristics similar to the underlying universe with the added benefit of improved index level
ESG performance.

## Approach

### Exploratory Data Analysis

During this phase, we identified and addressed data errors, uncovered various patterns, detected outliers, and discovered relationships among the variables.

### Interpolating missing values

During our Exploratory Data Analysis (EDA), we found 100 consecutive missing values. Initially, we attempted to interpolate these values using various methods, including linear, polynomial, and cubic spline functions. However, we found that calculating these values based on the open and previous close values proved to be more accurate and reliable results.

### Data integrity check

We identified and corrected several rows where the values of one variable did not correspond with others. We also detected and resolved issues with duplicate rows and rows sharing multiple columns in common.

### Outlier detection

We explored commonly used outlier detection techniques such as: Interquartile Range (IQR), Local Outlier Factor (LOF), Median Absolute Deviation (MAD) and Isolation Forest to find and address data points that deviate significantly from the dataset.

### Model development

We implemented 3 models: XGBoost, ARIMA and LSTM with a lookback period of 30 days and managed to obtain a RMSE value of 7.07.