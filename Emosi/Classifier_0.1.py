# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:35:15 2020

@author: Ghazian
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import csv

#Baca file csv fitur
database = pd.read_csv('fitur_emosi_V7_100_abs.csv')

#Pisahkan data dan header
x = database.drop('Class', axis=1)
y = database['Class']

#bagi data menjadi data latih dan data uji
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, random_state = 71) #setting split

#list parameter untuk dicoba, proses mencari hiperparameter terbaik
tuned_parameters = [{'kernel': ['rbf'],
                     'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                     'C': [0.1 ,1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]

#cari hiperparameter terbaik untuk presisi maksimal
print("# Tuning hyper-parameters for %s" % 'precision')
print()

clf = GridSearchCV(
    SVC(), tuned_parameters, cv = 10, scoring='%s_macro' % 'precision'
)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std, params))
#    row = (mean, std, params)
#    with open ('parameter_grids_v7_100_abs.csv',
#               mode='a', newline = '') as params:
#        fitur_writer = csv.writer(params, delimiter = ',')
#        fitur_writer.writerow(row)
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(x_test)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print()
    
##buat classifier
#svclassifier = SVC(kernel='rbf', C = 10, gamma = 1) #construct classifier
#
###latih
##svclassifier.fit(x_train, y_train)
##
###uji
##y_pred = svclassifier.predict(x_test)
#
###print hasil
##print(confusion_matrix(y_test,y_pred))
##print(classification_report(y_test,y_pred))
#
##buat struktur stratified kfold
#skf = RepeatedStratifiedKFold(n_splits=10, n_repeats = 10, random_state = 73)
#
##nilai classifier
#scores = cross_val_score(svclassifier, x, y, cv = skf)
#print(scores)
#print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
#row = (scores, scores.mean(), scores.std() * 2)
#with open ('cv_result_avg.csv', mode='a', newline = '') as params:
#        fitur_writer = csv.writer(params, delimiter = ',')
#        fitur_writer.writerow(row)