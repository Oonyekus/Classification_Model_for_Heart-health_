# Prediction of heart attacks using classification models

## Table of Content
- [Overview](#overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Analysis](#data-analysis)
- [Result and Findings](#result-and-findings)
- [Recommendation](#recommendation)
- [Limitations](#limitations)
- [References](#references)

## Overview
---
In this analysis, I predicted heart attacks using classification models, specifically logistic regression, K-nearest neighbours (KNN) algorithms and Decision Tree. The goal was to create models that, from a variety of variables, properly estimate the risk of a heart attack. To do this, 1 created 5 logical regression models, 3 KNN algorithms and 1 tree with different combinations and a number of predictor variables, each representing different aspects of an individual's health profile.

## Data Sources
---
The 'Heart health' dataset (Mahad, 2024) was obtained from Kaggle with permission for use. The dataset provides insights into individuals' heart health by holding vital signs and demographic data. The dataset includes variables such as age, gender, blood pressure, cholesterol levels, and history of heart attacks, making it a valuable resource for understanding cardiovascular health trends and assessing factors related to heart disease risk.


## Tools
---
The primary tool used for this project is Python. 

## Data Preparation
---
Although I initially obtained  a clean dataset, I manually entered a few missing values for demonstration purposes, showcasing one of many ways to deal with missing data. Additionally, duplicate entries and outlier values were removed, resulting in a finalised dataset comprising 701 rows and 11 columns. Also, for this report, blood pressure was split into Systolic and Diastolic, with primary attention given to Systolic blood pressure which has been proven by research to be a better predictor of heart health risk (Strandberg & Pitkala, 2023).

## Exploratory Data Analysis
---
what qts were asked to find the trend in data


## Data Analysis 
---
some interesting code
``` py
print ('Kelly')
```


## Result and Findings
---
Logistic Regression
 * Models 1 and 2 demonstrated excellent performance with perfect test accuracy, utilising fewer variables.
 *  Model 3, although slightly lower in accuracy, still performed well with a reduced set of variables.
 *  Models 4 and 5 also showed promising results, suggesting robustness across various variable combinations.

Recommendation: Considering the simplicity and interpretability of logistic regression, Models 1 and 2 could be preferred for deployment.


KNN
All KNN models showed high accuracy and robustness, with Model 2 performing slightly better than the others. KNN models are suitable when the dataset is not too large and computational efficiency is not a primary concern.
* Model 2 (KNN, k=5) might be the best choice due to its balanced performance and utilisation of a moderate number of variables.


Tree
* The decision tree model demonstrates high accuracy on both the training and test datasets, indicating its effectiveness in predicting heart attacks based on the provided features.
* The model's perfect training accuracy suggests that it may be overfitting to the training data. However, the high-test accuracy indicates that it generalises well to unseen data.


## Recommendation
---
Overall, logistic regression, Tree and KNN models offer reliable predictive performance for heart attack prediction. The choice between them depends on factors such as model interpretability, computational efficiency, and the specific requirements of the business or application. In this case, logistic regression appears to be the best choice for the following reasons:

1. Model Interpretability:
* Logistic regression provides clear and interpretable coefficients for
each predictor variable, allowing medical professionals to understand
the impact of each variable on the likelihood of a heart attack.
The coefficients can be directly related to odds ratios, making it easier
to communicate findings to stakeholders.


2. Computational Efficiency:
* Logistic regression is computationally efficient, especially when dealing with large datasets.
* Training and inference times are generally faster compared to more complex models like KNN, especially as the dataset size increases (PythonKitchen, 2024).


3. Specific Requirements of the Business or Application:
* In medical applications, interpretability and understanding of the model's decisions are crucial for gaining the trust of healthcare professionals and patients.
* Logistic regression's transparent nature and easily interpretable results align well with the need for transparency and trust in medical decision-making.
* Additionally, logistic regression can handle categorical variables and interactions between variables effectively, which are common in medical datasets. Based on this information and considerations, Model 2 of logistic regression appears to be the best model among all the options evaluated. Here is why...


4. High Accuracy:
* Model 2 achieved perfect accuracy on both the training and test datasets, indicating robust performance in predicting heart attacks.


5. Balanced Sensitivity and Specificity:
* Model 2 exhibited high sensitivity and specificity on both the training and test datasets, ensuring that it can effectively identify both positive and negative cases of heart attacks without sacrificing performance on either end.


6. Optimal Number of Variables:
* Model 2 utilised a relatively smaller number of variables (5) indicating a good balance between complexity and performance. This can help in maintaining model simplicity and interoperability.

7. Consistency:
* Model 2 consistently performed well across various evaluation metrics, indicating its stability and reliability in predicting heart attacks.

Therefore, considering its high accuracy, balanced sensitivity and specificity, optimal number of variables, and overall consistency, Model 2 of logistic regression stands out as the best choice among the evaluated models.

In the medical field, high accuracy and trust in predictive models are paramount. Accurate predictions ensure that patients receive timely interventions, aiding in diagnosis, treatment, and preventive measures. Trust in these models is essential for healthcare professionals to confidently rely on them for clinical decision-making, resource allocation, and risk management (Toma & Wei, 2023). Given these factors, the selection of Model 2 as the best model underscores its ability to provide accurate predictions, which is essential for enhancing patient care, supporting clinical decision- making, optimising resource utilisation, and mitigating health risks in the medical domain.


## Limitations 
---
what limitations, modification, removal, exclusion were done because of their effect on the analysis 


## References
---
