# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:18:30 2020

@author: Ghazian
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

database = pd.read_csv('fitur_emosi_all_abs.csv')

x = database.drop('Class', axis=1)

y = database['Class']

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                     'C': [0.1 ,1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]

skf = RepeatedStratifiedKFold(n_splits=10, n_repeats = 10, random_state = 73)

clf = GridSearchCV(SVC(), tuned_parameters, scoring= 'precision', cv = skf)

clf.fit(x, y)

#print(clf.cv_results_)