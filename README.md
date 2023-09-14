# Business Loan Credit Worthiness Prediction using an Optimized Neural Network Model

# Abstract

# Data Feature Engineering
## About the Data
The raw data used for this project is from the U.S. Small Business Administration (SBA) and was downloaded from Kaggle.com. The dataset include 27 features and 899,164 rows. The table below summarizes the features contents:

![image](https://github.com/wgemba/loan_default_prediction/assets/134420287/fe388d2e-0330-4293-9a83-6ac8b915fd54)

## Feature Engineering

The data set required significant preprocessing and feature engineering. Following several layers of preprocessing and engineering, including: transforming categorical features into numeric one, encoding large-set categorical features, removing class imbalance, removing null values, etc., the finalized dataset contains 81 features and 189,019 rows (see feature descriptions below).

![image](https://github.com/wgemba/loan_default_prediction/assets/134420287/04d8dba7-b2ce-45b9-a4fc-62e86212f367)

The full code for the feature processing can be seen in the file 'feature_engineering_and_visualization.ipynb'.

### Specific Packages Used
Certain features required the use of special packages and functions to convert from categorical to numeric datatype. Due to the large set of unique values for the feature 'Zip', rather than encoding the attribute I used the package 'pgeocode'. The package 'pgeocode' is a Python library that queries geospatial data (e.g., GPS coordinates, region name, municipality name) from the GeoNames database, using a postal code as input. Using this function, I created two new attributes 'latitude' and 'longitude' inplace of the 'Zip' feature. 

The 'State' attribute and and the 'Industry' attribute (which was created based on a dictionary remapping of the NAICS attributed, using the official NAICS code dictionary) were one-hot-encoded using the OneHotEncoder function from the Scikit-Learn preprocessing package.

## Neural Network Prediction Model Optimization and Model Evaluation.

