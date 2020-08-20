#Predicting Boston House-Prices

from sklearn.datasets import load_boston
import numpy as np

X, y = load_boston(return_X_y=True)

print(X.shape, y.shape)
#Out: ((506, 13), (506,))
print(np.min(y), np.max(y))
#Out: (5.0, 50.0)
