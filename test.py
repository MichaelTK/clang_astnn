#!/usr/bin/env python

import os, sys
import numpy as np

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

if __name__ == "__main__" :
    
    # Load Training
    X_train = []
    Y_train = []
    D_train = []
    fvector_labels = []

    for s in range(10):
        s1 = {'F1': 0, 'F2': 2000, 'F3': 10, 'F4': 4, 'F5': s}
        s2 = {'F2': 12, 'F3': 5, 'F4': 4, 'F5': s*2}
        s3 = {'F1': 5, 'F2': 21, 'F3': 2, 'F5': s*5}

        D_train.append(s1)
        Y_train.append('good')
        D_train.append(s2)
        Y_train.append('bad')
        D_train.append(s3)
        Y_train.append('good')

    v_train = DictVectorizer(sparse=False)
    X_train = v_train.fit_transform(D_train)  

    fvector_labels = v_train.feature_names_ 
    class_names = list(set(Y_train))

    print class_names
    print fvector_labels
    print X_train.shape
    print X_train


    # Load Testing
    X_test = []
    Y_test = []
    D_test = []

    for s in range(10):
        s1 = {'F2': 2000, 'F3': 5, 'F4': 40, 'F5': s, 'F900':10, 'F99':19}
        s2 = {'F2': 2, 'F3': 8, 'F4': 40, 'F5': s*2, 'F900':9, 'F99':10}
        s3 = {'F2': 2, 'F3': 7, 'F5': s*5, 'F100':9}

        D_test.append(s1)
        Y_test.append('good')
        D_test.append(s2)
        Y_test.append('bad')
        D_test.append(s3)
        Y_test.append('good')

        print '----',  v_train.transform(s1)[0]
        print '----',  v_train.transform(s2)[0]
        print '----',  v_train.transform(s3)[0]

    X_test = v_train.transform(D_test)  

    print X_test.shape
    print X_test

    # ------ Preprocess data 
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    std_scale = preprocessing.StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)

    print '------ Preprocess data StandardScaler'
    print X_train.shape
    print X_train
    print X_test.shape
    print X_test


    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    std_scale = preprocessing.RobustScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)


    print '------ Preprocess data'
    print X_train.shape
    print X_train
    print X_test.shape
    print X_test
