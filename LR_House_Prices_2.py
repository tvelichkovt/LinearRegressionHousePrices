# 1. Fetching the dataset
from sklearn.datasets import fetch_california_housing
# 2. Fetch the dataset into the variable dataset
dataset = fetch_california_housing()
# 3. Let’s look at the features of the dataset i.e the features of a particular house
feature_names = dataset['feature_names']
print("Feature names: {}\n".format(feature_names))
# 4. Feature Normalization
print(dataset.data)
from sklearn import preprocessing
data_original = (dataset.data)
X_scaled = preprocessing.scale(dataset.data)
# 5. Generating Polynomial features
from sklearn.preprocessing import PolynomialFeatures
pft = PolynomialFeatures(degree = 2)
X_poly = pft.fit_transform(X_scaled)
# 6. Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, dataset.target,test_size = 0.40,random_state = 42)
# 7. Fit a linear regression model
from sklearn import linear_model
model = linear_model.Ridge(alpha = 300) # alpha is the regularization parameter
model.fit(X_train, y_train)
# 8. Prediction on the test set
predictionTestSet = model.predict(X_test)
# 9. Error in prediction
from sklearn.metrics import mean_squared_error
errorTestSet = mean_squared_error(y_test, predictionTestSet) ; errorTestSet #0.53
