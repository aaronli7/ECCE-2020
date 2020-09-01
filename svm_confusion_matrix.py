#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 08:52:00 2020

@author: aran-lq
"""

import data_loader
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
import pickle
import numpy as np

from sklearn.metrics import plot_confusion_matrix

from mlxtend.plotting import plot_confusion_matrix as pcm


#%%
# knn = [[486   2   0]
 # [ 15  57   0]
 # [  4   0  27]]
 
# svm =array([[486,   2,   0],
#        [ 15,  57,   0],
#        [  4,   0,  27]])

all_path =  '../dataset/clean_data/allData/'
all_data = data_loader.PMUdataset(all_path)

ml_X = all_data.x_data
ml_X = ml_X.reshape(ml_X.shape[0], -1)
ml_Y = all_data.y_data
    
ml_X_train, ml_X_test, ml_Y_train, ml_Y_test = train_test_split(ml_X, ml_Y, test_size=0.15) 

knn_model = KNeighborsClassifier(n_neighbors = 50).fit(ml_X_train, ml_Y_train)
knn_pred = knn_model.predict(ml_X_test)
knn_acc = knn_model.score(ml_X_test, ml_Y_test)

dips = plot_confusion_matrix(knn_model, ml_X_test, ml_Y_test, cmap='Accent')

print(dips.confusion_matrix)
plt.show()



#%%
class_names = ['normal', 'DCDC_fault', 'VSC_delay']

#%%
svm_model = SVC(kernel = 'sigmoid', C = 1).fit(ml_X_train, ml_Y_train)
dips_svm= plot_confusion_matrix(knn_model, ml_X_test, ml_Y_test, cmap='Accent')


svm_matrix = np.array([[486, 1, 0],[12, 60, 0], [4, 0, 27]])