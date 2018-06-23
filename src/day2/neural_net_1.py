#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 18:49:10 2018

@author: jino
"""

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
'''
load dataset
'''
iris=load_iris()

#%%

'''
features
'''
data_features=iris.data
featurenames=iris.feature_names
#%%
'''
target
'''
target=iris.target
targetnames=iris.target_names
x=iris.data
y=iris.target

#%%
'''
visualize the dataset
'''
from matplotlib import pyplot as plt
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
x_index = 0
y_index = 1
plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2],format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.tight_layout()
plt.show()

#%%

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(10), learning_rate='constant',
       learning_rate_init=0.1, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=10,
       warm_start=False)

mlp.fit(X_train,y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))

print("Test set score: %f" % mlp.score(X_test, y_test))
predictions = mlp.predict(X_test)

#%%
'''
confusion matrix
'''
from sklearn.metrics import classification_report,confusion_matrix
import csv

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

cm=confusion_matrix(y_test,predictions)
with open("/home/tech/Desktop/Workshop/confusion_iris_dataset.csv","wb") as f:
    writer=csv.writer(f)
    writer.writerows(cm)

cm=classification_report(y_test,predictions)
with open("/home/tech/Desktop/Workshop/classification_report_iris_data.csv","wb") as f:
    writer=csv.writer(f)
    writer.writerows(cm)


#%%



