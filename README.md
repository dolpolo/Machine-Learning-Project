# Machine-Learning-Project

## A Comparison Between Several Approaches to Forecasting Using Many Predictors

### Replication Files for the Paper
**“A Comparative Study of Machine Learning Models for Forecasting Key Macroeconomic Indicators: How COVID-19 Betrayed Expectations”**

Authors:  
- **Chiara Perugini**, University of Bologna  
- **Davide Delfino**, University of Bologna  
- **Mohammad Kouzehgar Kaleji**, University of Bologna  
- **Sara Tozzi**, University of Bologna  

---

## Overview
This project trains and compares the forecasting performances of alternative forecasting methods for high-dimensional macroeconomic data. The main R scripts involved are:  
- `Bayesian_Shrinkage.R`  
- `FarmSelect.R`  

The outputs include:
1. Factor model forecasts (**Principal Components Regression**, PCR).  
2. Bayesian regression with i.i.d. normal prior (**Ridge Regression**).  
3. Bayesian regression with i.i.d. Laplace prior (**LASSO Regression**).  
4. Factor-Adjusted model (**FarmSelect**).

The data used in this project were retrieved from:  
**Barigozzi, M., & Lissona, C. (2024)**: *EA-MD-QD: Large Euro Area and Euro Member Countries Datasets for Macroeconomic Research (Version 12.2023)*.  
[Zenodo DOI: 10.5281/ZENODO.10514668](https://doi.org/10.5281/ZENODO.10514668).

---

## Scripts

### Main Scripts
- **`Bayesian_Shrinkage.R`**: Trains and compares forecasting methods for PCR, Ridge, and LASSO.  
- **`FarmSelect.R`**: Implements the Factor-Adjusted Regression Model (FarmSelect).  
- **`Forecasting_methods.R`**: Produces out-of-sample forecasts for all models.  
- **`Covid_predictions.R`**: Produces counterfactual forecasts for the best-performing models during the COVID-19 pandemic (2020).  

### Supporting Scripts in `R/functions`:
- **`Bayesian_shrinkage_functions.R`**: Defines parameters and computes forecasts for PCR, LASSO, and Ridge regression.  
- **`FarmSelect_functions.R`**: Defines parameters and computes forecasts for FarmSelect.

---

## Instructions

### Loading Data
Data is stored in the `data/EA-MD-QD/` folder.  
Use the script `Bayesian_Shrinkage.R` to load and preprocess the data, ensuring the correct working directory is set.  

### Troubleshooting `source()`:
If the `source()` function does not load the R functions correctly, manually run each function in the `R/functions` folder before executing the main scripts.

---

## Additional Scripts
- **`Descriptive_analysis.R`**: Provides a descriptive analysis of the dataset.

---

## How to Run
1. Clone this repository to your local machine.
2. Set the working directory to the project folder:
   ```R
   path <- "C:/Users/Davide/Desktop/Alma Mater/SECOND YEAR/Machine Learning/Machine-Learning-Project"
   setwd(path)

