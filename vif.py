# Detecting Multicollinearity using VIF
import pandas as pd
df = pd.read_csv('salary_small.csv') ; print(df.head())

# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

# calc_vif(df)
print(calc_vif(df)) ; #experience and age are over-correlated >> 5 !!!

# FIX calc_vif(df)
X = df.drop(['exp','age'],axis=1)
print(calc_vif(X)) # df.to_csv('out.csv')
