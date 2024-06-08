# Saving Machine Learning Model

This repository demonstrates how to train a simple linear regression model using scikit-learn and then save and load the trained model using both `pickle` and `joblib`.

## Overview

The dataset used is the Canada Per Capita Income dataset, which contains information on per capita income over several years. The project includes the following steps:

1. Importing libraries
2. Reading the dataset
3. Plotting a scatter plot for data visualization
4. Training a linear regression model
5. Making predictions using the trained model
6. Saving the trained model using `pickle`
7. Saving the trained model using `joblib`
8. Loading and using the saved models to make predictions

## Installation

To run this project, you need to install the following Python packages:

- pandas
- numpy
- matplotlib
- scikit-learn
- pickle (included in Python's standard library)
- joblib (included in scikit-learn)

You can install the required packages using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

### 1. Importing Libraries

The necessary libraries are imported to handle data operations, visualization, and machine learning.

### 2. Reading the Dataset

The dataset is read using pandas:

```python
df = pd.read_csv('canada_per_capita_income.csv')
```

### 3. Plotting Scatter Plot

A scatter plot of the data is created to visualize the relationship between the year and per capita income:

```python
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.title('Per Capita Income Over Years')
plt.scatter(x=df['year'], y=df['per capita income (US$)'], color='red', marker='+')
```

### 4. Training the Linear Regression Model

A linear regression model is trained using the dataset:

```python
reg = linear_model.LinearRegression()
reg.fit(df[['year']], df[['per capita income (US$)']])
```

### 5. Making Predictions

The trained model is used to make predictions:

```python
reg.predict([[2024]])
```

### 6. Saving the Model using `pickle`

The trained model is saved and loaded using `pickle`:

```python
import pickle

with open('model_pickle', 'wb') as file:
    pickle.dump(reg, file)

with open('model_pickle', 'rb') as file:
    model = pickle.load(file)

model.predict([[2024]])
```

### 7. Saving the Model using `joblib`

The trained model is saved and loaded using `joblib`:

```python
import joblib

joblib.dump(reg, 'model_joblib')

joblib_model = joblib.load('model_joblib')

joblib_model.predict([[2024]])
```

### 8. Model Coefficients and Intercept

The model's coefficients and intercept can be accessed as follows:

```python
reg.coef_
reg.intercept_

joblib_model.coef_
joblib_model.intercept_
```

## Conclusion

This project demonstrates the process of training a linear regression model and saving it using both `pickle` and `joblib`. These techniques are useful for preserving models and reusing them without retraining, which can save time and computational resources.

