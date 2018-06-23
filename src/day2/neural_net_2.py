#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 19:25:28 2018

@author: root
"""

from sklearn.datasets import load_digits

from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
import numpy as np

digits=load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

    
'''
test=images_and_labels[:4]
print test[0][0]
print test[0][1]
plt.imshow(test[0][0],cmap=plt.cm.gray)
'''    
#%%
'''
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, 64))
data_0=np.reshape(data[0],(8,8))
plt.imshow(test[3][0],cmap=plt.cm.gray)
'''
data = digits.images.reshape((n_samples, 64))
x=data[:n_samples]
y=digits.target[:n_samples]

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100), learning_rate='constant',
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
with open("/home/tech/Desktop/Workshop/confusion_digit_dataset.csv","wb") as f:
    writer=csv.writer(f)
    writer.writerows(cm)

cm=classification_report(y_test,predictions)
with open("/home/tech/Desktop/Workshop/classification_report_digit_data.csv","wb") as f:
    writer=csv.writer(f)
    writer.writerows(cm)

#%%

import math
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]
print([round(i, 2) for i in z_exp])
sum_z_exp = sum(z_exp)
print(round(sum_z_exp, 2))
softmax = [round(i / sum_z_exp, 3) for i in z_exp]
print(softmax)