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
In this analysis, I predicted heart attacks using classification models, specifically logistic regression, K-nearest neighbours (KNN) algorithms, and Decision Tree. The goal was to create models that, from a variety of variables, properly estimate the risk of a heart attack. To do this, 1 created 5 logical regression models, 3 KNN algorithms, and 1 tree with different combinations and a number of predictor variables, each representing different aspects of an individual's health profile.

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
To perform a comprehensive EDA, these questions were highlighted as the guide.
* What is the overall structure and summary statistics of the dataset?
* Are there any missing values or outliers in the dataset, and how should they be handled?
* What is the distribution of the target variable (heart attack occurrence), and is there any class imbalance?
* What are the relationships between the predictor variables and the target variable?
* How are the predictor variables correlated with each other, and are there any multicollinearity issues?

## Data Analysis 
---
some interesting code
``` py
Heart_health[['Systolic BP', 'Diastolic BP']] = Heart_health['Blood Pressure(mmHg)'].astype(str).str.split('/', expand=True)
#Heart_health[['Systolic BP', 'Diastolic BP']] = Heart_health['Blood Pressure'].apply(split_blood_pressure).apply(pd.Series)
Heart_health.drop(columns=['Blood Pressure(mmHg)'], inplace=True)
```

``` py
Heart_health[['Systolic BP', 'Diastolic BP']] = Heart_health[['Systolic BP', 'Diastolic BP']].astype(int)
Heart_health['Gender'] = Heart_health['Gender'].map({'Male': 1, 'Female': 0})
Heart_health['Smoker'] = Heart_health['Smoker'].map({'No': 0, 'Yes': 1 })
print(Heart_health.dtypes)
```

``` py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Selecting predictors by dropping the columns not needed
X = Heart_health.drop(['Height(cm)', 'Smoker','Diastolic BP'], axis=1)

# Target variable
Y = Heart_health['Heart Attack']

# Splitting the dataset into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Creating and fitting the logistic regression model
model_1= LogisticRegression()
model_1.fit(X_train, Y_train)

# Making predictions on both the training and test sets
Y_train_pred = model_1.predict(X_train)
Y_test_pred = model_1.predict(X_test)

# Calculating and printing performance metrics
print("Training Accuracy:", accuracy_score(Y_train, Y_train_pred))
print("Test Accuracy:", accuracy_score(Y_test, Y_test_pred))

print("\nConfusion Matrix (Train Data):")
print(confusion_matrix(Y_train, Y_train_pred))

print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(Y_test, Y_test_pred))

cm_train = confusion_matrix(Y_train, Y_train_pred)

# Extract values from confusion matrix for training data
TN_train = cm_train[0, 0]
FP_train = cm_train[0, 1]
FN_train = cm_train[1, 0]
TP_train = cm_train[1, 1]

# Calculate sensitivity (recall) and specificity for training data
sensitivity_train = TP_train / (TP_train + FN_train)
specificity_train = TN_train / (TN_train + FP_train)

print("\nTraining Sensitivity:", sensitivity_train)
print("Training Specificity:", specificity_train)

# Calculate confusion matrix for test data
cm_test = confusion_matrix(Y_test, Y_test_pred)

# Extract values from confusion matrix for test data
TN_test = cm_test[0, 0]
FP_test = cm_test[0, 1]
FN_test = cm_test[1, 0]
TP_test = cm_test[1, 1]

# Calculate sensitivity (recall) and specificity for test data
sensitivity_test = TP_test / (TP_test + FN_test)
specificity_test = TN_test / (TN_test + FP_test)

print("\nTest Sensitivity:", sensitivity_test)
print("Test Specificity:", specificity_test)
```

``` py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# This is Model 1 ( (with KNN = 3)

# Selecting predictors
X = Heart_health[['Weight(kg)', 'Cholesterol(mg/dL)','Exercise(hours/week)' ]]
# Target variable
Y = Heart_health['Heart Attack']

# For some models like KNN, we need to do feature scaling
scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
scaler.fit(X)
X_std = scaler.transform(X)

# Splitting the dataset into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Creating and fitting the KNN model with k=3
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, Y_train)

# Making predictions on both the training and test sets
Y_train_pred_knn = knn_model.predict(X_train)
Y_test_pred_knn = knn_model.predict(X_test)

# Calculating and printing performance metrics
print("Training Accuracy (KNN, k=3):", accuracy_score(Y_train, Y_train_pred_knn))
print("Test Accuracy (KNN, k=3):", accuracy_score(Y_test, Y_test_pred_knn))

print("\nConfusion Matrix (Train Data, KNN, k=3):")
print(confusion_matrix(Y_train, Y_train_pred_knn))

print("\nConfusion Matrix (Test Data, KNN, k=3):")
print(confusion_matrix(Y_test, Y_test_pred_knn))


cm_train = confusion_matrix(Y_train, Y_train_pred_knn)

# Extract values from confusion matrix for training data
TN_train = cm_train[0, 0]
FP_train = cm_train[0, 1]
FN_train = cm_train[1, 0]
TP_train = cm_train[1, 1]

# Calculate sensitivity (recall) and specificity for training data
sensitivity_train = TP_train / (TP_train + FN_train)
specificity_train = TN_train / (TN_train + FP_train)

print("\nTraining Sensitivity:", sensitivity_train)
print("Training Specificity:", specificity_train)

# Calculate confusion matrix for test data
cm_test = confusion_matrix(Y_test, Y_test_pred_knn)

# Extract values from confusion matrix for test data
TN_test = cm_test[0, 0]
FP_test = cm_test[0, 1]
FN_test = cm_test[1, 0]
TP_test = cm_test[1, 1]

# Calculate sensitivity (recall) and specificity for test data
sensitivity_test = TP_test / (TP_test + FN_test)
specificity_test = TN_test / (TN_test + FP_test)

print("\nTest Sensitivity:", sensitivity_test)
print("Test Specificity:", specificity_test)
```

``` py
# List of numerical variables
numerical_vars = ['Age', 'Height(cm)', 'Weight(kg)', 'Cholesterol(mg/dL)', 'Glucose(mg/dL)', 'Exercise(hours/week)', 'Systolic BP', 'Diastolic BP']

# Setting up the figure and axes for the 3x3 grid
fig, axes = plt.subplots(1, 2, figsize=(8, 5)) # Adjust the size as needed
fig.suptitle('Relationship between Numerical Variables and Heart Attack')

#axes = axes.flatten()  # Flattening the array of axes for easy iteration
sns.boxplot(x='Heart Attack', y= 'Age', data=Heart_health, ax=axes[0])
axes[0].set_title(f'{numerical_vars[0]} vs Heart Attack', fontsize=14)
axes[0].set_xlabel('') # Remove x labels to declutter
axes[0].set_ylabel('')

sns.boxplot(x='Heart Attack', y= 'Height(cm)', data=Heart_health, ax=axes[1])
axes[1].set_title(f'{numerical_vars[1]} vs Heart Attack', fontsize=14)
axes[1].set_xlabel('') # Remove x labels to declutter
axes[1].set_ylabel('')

# categorical variables
categorical_vars = ['Gender', 'Smoker']

# Setting up the figure for the 1x3 grid
fig, axes = plt.subplots(1, 2, figsize=(18, 6)) # Adjust the size as needed

# Manually creating each plot
sns.countplot(x='Gender', hue='Heart Attack', data=Heart_health, ax=axes[0])
axes[0].set_title('Distribution of Heart Attack by Gender')

sns.countplot(x='Smoker', hue='Heart Attack', data=Heart_health, ax=axes[1])
axes[1].set_title('Distribution of Heart Attack by Smoker')

plt.tight_layout()  # Adjust the layout to make room for the titles
plt.show()
```

``` py
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Selecting predictors by dropping 'Sales' and target variable 'High'
X = Heart_health.drop(['Heart Attack'], axis=1)
X
# Encoding categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)  # drop_first to avoid dummy variable trap
X_encoded

# Target variable
Y = Heart_health['Heart Attack']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

# Creating and fitting the Decision Tree model on the training dataset
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=4) ##### Increasing max depth gives better training accuracy, but lower test accuracy

model.fit(X_train, Y_train)

# Making predictions on both the training and test sets
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Calculating and printing performance metrics
print("Training Accuracy:", accuracy_score(Y_train, Y_train_pred))
print("Test Accuracy:", accuracy_score(Y_test, Y_test_pred))

print("\nConfusion Matrix (Train Data):")
print(confusion_matrix(Y_train, Y_train_pred))

print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(Y_test, Y_test_pred))

cm_train = confusion_matrix(Y_train, Y_train_pred)

# Extract values from confusion matrix for training data
TN_train = cm_train[0, 0]
FP_train = cm_train[0, 1]
FN_train = cm_train[1, 0]
TP_train = cm_train[1, 1]

# Calculate sensitivity (recall) and specificity for training data
sensitivity_train = TP_train / (TP_train + FN_train)
specificity_train = TN_train / (TN_train + FP_train)

print("\nTraining Sensitivity:", sensitivity_train)
print("Training Specificity:", specificity_train)

# Calculate confusion matrix for test data
cm_test = confusion_matrix(Y_test, Y_test_pred)

# Extract values from confusion matrix for test data
TN_test = cm_test[0, 0]
FP_test = cm_test[0, 1]
FN_test = cm_test[1, 0]
TP_test = cm_test[1, 1]

# Calculate sensitivity (recall) and specificity for test data
sensitivity_test = TP_test / (TP_test + FN_test)
specificity_test = TN_test / (TN_test + FP_test)

print("\nTest Sensitivity:", sensitivity_test)
print("Test Specificity:", specificity_test)
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
