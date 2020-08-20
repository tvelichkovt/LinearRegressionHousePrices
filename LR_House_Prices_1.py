# === Predicting Boston House-Prices ===

# 1. Load Data

from sklearn.datasets import load_boston
import numpy as np

X, y = load_boston(return_X_y=True)

# 2. Univariate Model using Least Squares == Heteroskedasticity
#Types of variables
    #Categorical: Takes distinct values: Spam/Not Spam email, Diabetes +ve/-ve
    #Continuous: Can take infinite values, e.g. money, time, weight.
    #Dependent: Outcome of the experiment, ex: house values
    #Independent: Variable independent of what the researcher does. Area, location, #bedrooms etc.

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

X = X[:, np.newaxis, 4]

x_train, x_test = (X[:400], X[400:])
y_train, y_test = (y[:400], y[400:])

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print(mean_squared_error(y_pred, y_test))
#Output: 54.926

print(reg.coef_, reg.intercept_)
#Output: array([-24.408]), 37.274

x1 = np.linspace(np.min(x_test), np.max(x_test))
y1 = reg.coef_*x1 + reg.intercept_

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(x_test, y_test, label='Data Points')
plt.plot(x1, y1, linewidth=2.5, color='r', label='Best Fit Line')
plt.xlabel('X values for 5th feature')
plt.ylabel('House prices')
plt.title('Linear regression with One Feature')
plt.legend()


# 3. Multivariate Least Squares Method

y_pred = reg.predict(x_test)
print(mean_squared_error(y_pred, y_test))
#Output: 37.894

print(reg.coef_, reg.intercept_)

# 4. Variance Inflation Factor(VIF) vs == multicollinearity ==
    # VIF greater than 5.0 indicates a highly correlated variable. Therefore, those highly correlated variables must be removed for a good prediction model.

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.DataFrame(X)
df.dropna()
df = df._get_numeric_data() #This line will drop non-numeric cols

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

print(vif)









