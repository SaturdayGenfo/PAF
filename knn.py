# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 21:31:52 2017


"""
from sklearn.neighbors import KNeighborsClassifier
from dataformat import mnist
import numpy as np

X, Y = mnist.train.next_batch(15000)

classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X,Y)

testX, testY = mnist.test.next_batch(1000)

results = classifier.predict(testX)

accuracy = sum([(np.argmax(results[i]) == np.argmax(testY[i]) )for i in range(len(testY))])
print(accuracy*1.0/len(testY))
