# pandemic_handling_and_impact

## Overview
Covid -19 has affected the majority of our livelihoods in the past 2.5 years. In fact, every once in a while we suffer from pandemics that threaten life on earth. Usually, we tend to ignore such situations until we have been directly or indirectly affected. We assume that the healthcare industry will develop a vaccine, and the government will create suitable policies to localize the spread, but usually, the handling could be smoother. In this project, we aim to answer questions that pertain not to COVID in terms of weekly/monthly statistics but as a Pandemic. We aim to analyze and comment on the psychological impact that the pandemic of COVID-19 has had on the world. Using Explainable AI, we extract the importance of factors contributing to stress and give an overview of what a country needs to improve on, to better handle its next pandemic.

## Workflow

![Project Workflow](./Workflow.png?raw=true)

## EDA
1. EDA.ipynb <br  />
  **Datasets Used:**   <br  />
    a. COVIDiSTRESS_April_27.csv <br  />
    b. COVIDiSTRESS_May_30.csv <br  />
    c. COVIDiSTRESS_June_17.csv <br  />
  **EDA Methods:** <br  />
    a. Correlation Analysis <br  />
    b. Pairplots <br  />
    c. Distribution Analysis (Age) <br  />
    d. Histograms <br  />
2. EDA.twb <br  />
  **Datasets Used:** <br  />
     a. COVIDiSTRESS_April_27.csv <br  />
  **EDA Methods:** <br  />
    a. Age Distribution - histogram <br  />
    b. Employment Status - Pie Chart <br  />
    c. Gender Distribution - Count Plot <br  />
    d. Education Distribution - Pie Chart <br  />
    e. Geographical Distribution - World Map <br  />
    f. Gender vs Stress - Heatmap <br  />
    g. Age vs Stress - Heatmap <br  />
    h. Loneliness vs Stress - Line Plot <br  />

## Data Cleaning & Feature Engineering
1. data_preparation.py <br  />
2. scaling.ipynb <br  />

## Models & Results
1. model.ipynb - Deep neural network model <br  />
2. shap.ipynb - SHAP implementation for extracting feature importances <br  />
3. XGBoost.ipynb - XGBoost model and feature importance plot <br  />
4. XGBoost.html  <br  />

## Utilities & Resources
1. country-codes-lat-long-alpha3.json - json file containing following information for all countries: <br  />
  a. ISO 3166-1 alpha-3 codes <br  />
  b. longitude and latitude (average) <br  />
  c. numeric codes <br  />
  d. country names <br  />
  e. alpha-2 codes <br  />
2. var/ - pickle dumps of encoded/scaled features <br  />
  a. age_scaler.pkl <br  />
  b. ohe_empl.pkl <br  />
  c. ohe_isol.pkl <br  />
  d. ohe_marital.pkl <br  />
3. resources/ - SHAP feature importance (images) <br  />
  a. countries/ <br  />
  b. world_importance_nn.png <br  />

## Analysis & Visualization
1. world_stress_heatmap.ipynb - World Map with Average Stress Values Plotted for Each Country <br  />
2. Important_Features.ipynb - Pandemic Preparedness: Primary Source of Stress for Each Country <br  />
