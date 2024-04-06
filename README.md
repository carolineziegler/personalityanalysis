# **University Demographics and Personalities Analysis**

This repository contains Python scripts designed to analyze and model demographic and personality data of students at Dublin City University (DCU). The project is structured to clean the data, perform exploratory data analysis, and build predictive models based on the demographics and personality traits collected through surveys.
Project Overview

The project involves the following key components:

Data Cleaning: Processing raw survey data to correct anomalies and prepare it for analysis.
Exploratory Data Analysis (EDA): Analyzing the cleaned data to discover patterns and trends, and visualizing these findings using Python's matplotlib library.
Predictive Modeling: Using linear regression to predict various outcomes based on demographic and personality data.
Statistical Testing: Performing t-tests to explore differences in personality traits across different demographic groups.

## **Getting Started**
Prerequisites
Ensure you have Python installed on your system along with the following libraries:

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn

You can install these packages using pip if you haven't already:

  pip install pandas numpy matplotlib seaborn scikit-learn

## **Installation**

**Running the Scripts**

To run the main analysis script, execute the following command from the root directory of the project:

  python demographics_analysis.py

**Files and Directories**

    demographics.csv: Contains raw demographic data collected via surveys.
    personalities.csv: Contains raw personality trait data linked to the demographic data by participant ID.
    demographics_analysis.py: The main script that performs data cleaning, EDA, and statistical analysis.
    predictive_models.py: Contains regression models and predictive analysis.

## **Features and Tasks**

**Data Cleaning:**
- Handling missing values and data inconsistencies.
- Normalizing data formats and merging datasets on common identifiers.

**Exploratory Data Analysis:**
- Distribution of demographic variables such as age, gender, and location.
- Analysis of travel patterns to university and their correlation with postal codes.

**Predictive Modeling:**
- Linear regression models to predict outcomes such as exam results based on demographic and personality data.
- Validation of models using split datasets to ensure the reliability of the predictions.

**Statistical Testing:**
- Comparison of personality traits between different gender groups using t-tests.
- Analysis of personality traits across different zodiac signs.
