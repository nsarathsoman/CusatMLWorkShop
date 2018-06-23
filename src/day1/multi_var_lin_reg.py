train = data = np.loadtxt(open("/Users/Sarath/Projects/CusatMLWorkShop/data/day1/housing_data.csv", "rb"), \
  delimiter = ',')

X = train[:, 0:-1]
y = train[:, -1].reshape(-1, 1)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn import linear_model as lm

regr = lm.LinearRegression(fit_intercept=True)

regr.fit(X_train, Y_train)

Y_pred = regr.predict(X_test)

from sklearn.metrics import mean_squared_error

rms = mean_squared_error(Y_test, Y_pred)

import math
math.sqrt(rms)
