
import numpy as np

train = data = np.loadtxt(open("/Users/Sarath/Projects/CusatMLWorkShop/data/day1/data_train_sv.csv", "rb"), \
  delimiter = ',')

x_train = train[:, 0].reshape(-1, 1)
y_train = train[:, 1].reshape(-1, 1)

test = data = np.loadtxt(open("/Users/Sarath/Projects/CusatMLWorkShop/data/day1/data_test_sv.csv", "rb"), \
  delimiter = ',')

x_test = test[:, 0].reshape(-1, 1)
y_test = test[:, 1].reshape(-1, 1)

print data

#import model
from sklearn import linear_model as lm

#select model
regr = lm.LinearRegression(fit_intercept=True)

#Train the model using training data
regr.fit(x_train, y_train)

#make preductions using test data
y_pred = regr.predict(x_test)

regr.coef_
regr.intercept_

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)

import matplotlib.pyplot as plt
plt.scatter(x_test, y_test, color='black')