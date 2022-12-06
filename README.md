# pandemic_handling_and_impact

## Overview
Covid -19 has affected the majority of our livelihoods in the past 2.5 years. In fact, every once in a while we suffer from pandemics that threaten life on earth. Usually, we tend to ignore such situations until we have been directly or indirectly affected. We assume that the healthcare industry will develop a vaccine, and the government will create suitable policies to localize the spread, but usually, the handling could be smoother. In this project, we aim to answer questions that pertain not to COVID in terms of weekly/monthly statistics but as a Pandemic. We aim to analyze and comment on the psychological impact that the pandemic of COVID-19 has had on the world. Using Explainable AI, we extract the importance of factors contributing to stress and give an overview of what a country needs to improve on, to better handle its next pandemic.

## Workflow

![Project Workflow](https://github.com/Shaashwat05/pandemic_handling_and_impact]/blob/main/Workflow.png?raw=true)

## EDA
1. EDA.ipynb
  Datasets Used: 
    a. COVIDiSTRESS_April_27.csv
    b. COVIDiSTRESS_May_30.csv
    c. COVIDiSTRESS_June_17.csv
  EDA Methods:
    a. Correlation Analysis
    b. Pairplots
    c. Distribution Analysis (Age)
    d. Histograms
2. EDA.twb
  Datasets Used: 
     a. COVIDiSTRESS_April_27.csv
  EDA Methods: 
    a. Age Distribution - histogram
    b. Employment Status - Pie Chart
    c. Gender Distribution - Count Plot
    d. Education Distribution - Pie Chart
    e. Geographical Distribution - World Map
    f. Gender vs Stress - Heatmap
    g. Age vs Stress - Heatmap
    h. Loneliness vs Stress - Line Plot

## Data Cleaning & Feature Engineering
1. data_preparation.py
2. scaling.ipynb

## Models & Results
1. model.ipynb - Deep neural network model
2. shap.ipynb - SHAP implementation for extracting feature importances
3. XGBoost.ipynb - XGBoost model and feature importance plot
4. XGBoost.html 

## Utilities & Resources
1. country-codes-lat-long-alpha3.json - json file containing following information for all countries:
  a. ISO 3166-1 alpha-3 codes
  b. longitude and latitude (average)
  c. numeric codes
  d. country names
  e. alpha-2 codes
2. var/ - pickle dumps of encoded/scaled features
  a. age_scaler.pkl
  b. ohe_empl.pkl
  c. ohe_isol.pkl
  d. ohe_marital.pkl
3. resources/ - SHAP feature importance (images)
  a. countries/
  b. world_importance_nn.png

## Analysis & Visualization
1. world_stress_heatmap.ipynb - World Map with Average Stress Values Plotted for Each Country
2. Important_Features.ipynb - Pandemic Preparedness: Primary Source of Stress for Each Country
