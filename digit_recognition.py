# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:19:36 2018

@author: Anubhav2000
"""
import time
import mnist_loader as mnist
from sklearn import svm
import numpy

def svm_baseline():
    training_data, validation_data, test_data = mnist.load_data()
    # train
    t0=time.time()
    clf = svm.SVC(kernel='rbf',degree=4)
    from sklearn import preprocessing
    X_train=preprocessing.scale(training_data[0])
    scaler=preprocessing.StandardScaler().fit(training_data[0])
    clf.fit(X_train, training_data[1])
    print("Training time: ",time.time()-t0)
    # test
    t0=time.time()
    X_test=scaler.transform(test_data[0])
    predictions = [int(a) for a in clf.predict(X_test)]
    print("testing time:",time.time()-t0)
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print ("Baseline classifier using an SVM.")
    print ("%s of %s values correct." % (num_correct, len(test_data[1])))

if __name__ == "__main__":
    svm_baseline()
#######################
