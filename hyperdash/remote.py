#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/23 9:38
# @Author  : Aries
# @Site    : 
# @File    : remote.py
# @Software: PyCharm


from sklearn import svm,datasets
from hyperdash import Experiment

digits = datasets.load_digits()
test_cases = 50
X_train,y_train = digits.data[:-test_cases],digits.target[:-test_cases]
X_test,y_test = digits.data[-test_cases:],digits.target[-test_cases:]
exp = Experiment("Digits Classifier")
gamma = exp.param("gamma",0.01)
classifer = svm.SVC(gamma=gamma)
classifer.fit(X_train,y_train)
exp.metric("accuracy",classifer.score(X_test,y_test))
exp.end()