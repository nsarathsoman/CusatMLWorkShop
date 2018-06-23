import numpy as np
from sklearn.model_selection import train_test_split
from sklear import datasets

data = datasets.load_breast_cancer()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y.reshape(-1,1),
                                                    test_size=0.20, random_state=42)

from sklearn import linear_model as lm

classifier = lm.LogisticRegression()
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)