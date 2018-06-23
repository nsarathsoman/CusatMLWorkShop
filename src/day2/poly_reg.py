import numpy as np

X = np.linspace(0, 0.5, 100)

y_actual = np.cos(2 * np.pi * X) ** 2

noise = np.random.normal(0, 0.1, 100)

y = y_actual + noise

import matplotlib.pyplot as plt

plt.scatter(X, y, color='red')
plt.plot(X, y_actual, color="blue", linewidth=1)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1,1), y.reshape(-1,1),
                                                    test_size=0.20, random_state=42)

from sklearn import linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

deg = 10
polynomial_features = PolynomialFeatures(degree=deg, include_bias=True)
linear_regression = lm.LinearRegression()

pipeline = Pipeline([("polynomial_features",polynomial_features),\
                     ("linear_regression",linear_regression)])
pipeline.fit(X_train, Y_train)

Y_pred = pipeline.predict(X_test)

plt.scatter(X, y, color='red')
plt.plot(X_test, y_pred, color="blue", linewidth=1)
plt.show()