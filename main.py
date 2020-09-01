#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:58:33 2020

@author: aran-lq
"""
import data_loader
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib as mpl
mpl.use('Agg')
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
#from sklearn.model_selection import KFold

# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 32
TIME_STEP = 55  # length of LSTM time sequence
INPUT_SIZE = 18 # num of feature
LR = 0.01   # learning rate
KFOLD = 10
isGPU = torch.cuda.is_available()

#training_path = '../dataoutput/clean_data/training/'
#test_path = '../dataoutput/clean_data/test/'
all_path =  '../dataset/clean_data/allData/'

#training_data = data_loader.PMUdataset(training_path)
#test_data = data_loader.PMUdataset(test_path)
all_data = data_loader.PMUdataset(all_path)
## 15% testing, 85% training
#training_data, test_data = torch.utils.data.random_split(all_data, [3344, 591])
#
#training_Loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
#test_Loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

#----------------------create the LSTM Net ------------------------------------
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=2,
                batch_first=True,
                )
        #fully connected
        self.out = nn.Linear(32, 3)
    
    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(lstm_out[:, -1, :])
        return out

#-------------------create the CNN Net ----------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ), # -> (16, 53, 16)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, ceil_mode=False),
        )# -> (16, 27, 8)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 32, 5, 1, 2), # -> (16, 25, 6)
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # ) # -> (32, 13, 3)
        self.out = nn.Linear(16 * 27 * 9, 3)
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x) # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

#-------------------create the NN Net -----------------------------------------
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(55 * 18, 3),
            # nn.ReLU(),
#            nn.Linear(2 * 55 * 18, 3)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output 
    
#----------------------training -----------------------------------------
# data_metrics = {'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}
data_output = {'knn':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}, 
               'dtree':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}, 
               'svm':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}, 
               'lstm':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}, 
               'cnn':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}, 
               'ann':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[]}}
for num_of_training in range(KFOLD):
    
    
    #--------------three machine learning method-------------------------
    ml_X = all_data.x_data.numpy()
    ml_X = ml_X.reshape(ml_X.shape[0], -1)
    ml_Y = all_data.y_data.numpy()
    
    ml_X_train, ml_X_test, ml_Y_train, ml_Y_test = train_test_split(ml_X, ml_Y, test_size=0.15)
    
    # train dtree
    dtree_model = DecisionTreeClassifier(max_depth = 1).fit(ml_X_train, ml_Y_train)
    dtree_pred = dtree_model.predict(ml_X_test)
    dtree_acc = dtree_model.score(ml_X_test, ml_Y_test)
    
    # train SVM 
    svm_model = SVC(kernel = 'sigmoid', C = 1).fit(ml_X_train, ml_Y_train)
    svm_pred = svm_model.predict(ml_X_test)
    svm_acc = svm_model.score(ml_X_test, ml_Y_test)
    
    # train KNN 
    knn_model = KNeighborsClassifier(n_neighbors = 50).fit(ml_X_train, ml_Y_train)
    knn_pred = knn_model.predict(ml_X_test)
    knn_acc = knn_model.score(ml_X_test, ml_Y_test)
    
    # F1 and ROC 
    dtree_f1 = sklearn.metrics.f1_score(dtree_pred, ml_Y_test, average='weighted')
    dtree_precision = sklearn.metrics.precision_score(dtree_pred, ml_Y_test, average='weighted')
    dtree_recall = sklearn.metrics.recall_score(dtree_pred, ml_Y_test, average='weighted')
    
    svm_f1 = sklearn.metrics.f1_score(svm_pred, ml_Y_test, average='weighted')
    svm_precision = sklearn.metrics.precision_score(svm_pred, ml_Y_test, average='weighted')
    svm_recall = sklearn.metrics.recall_score(svm_pred, ml_Y_test, average='weighted')
    
    knn_f1 = sklearn.metrics.f1_score(knn_pred, ml_Y_test, average='weighted')
    knn_precision = sklearn.metrics.precision_score(knn_pred, ml_Y_test, average='weighted')
    knn_recall = sklearn.metrics.recall_score(knn_pred, ml_Y_test, average='weighted')
    
    svm_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2])
    svm_pred_y = label_binarize(svm_pred, classes=[0, 1, 2])
    svm_fpr, svm_tpr, _ = roc_curve(svm_test_y.ravel(), svm_pred_y.ravel())
    svm_roc_auc = sklearn.metrics.auc(svm_fpr, svm_tpr)
    
    knn_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2])
    knn_pred_y = label_binarize(knn_pred, classes=[0, 1, 2])
    knn_fpr, knn_tpr, _ = roc_curve(knn_test_y.ravel(), knn_pred_y.ravel())
    knn_roc_auc = sklearn.metrics.auc(knn_fpr, knn_tpr)
    
    dtree_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2])
    dtree_pred_y = label_binarize(dtree_pred, classes=[0, 1, 2])
    dtree_fpr, dtree_tpr, _ = roc_curve(dtree_test_y.ravel(), dtree_pred_y.ravel())
    dtree_roc_auc = sklearn.metrics.auc(dtree_fpr, dtree_tpr)
    
    # machine learning method output
    data_output['knn']['F1'].append(knn_f1)
    data_output['knn']['precision'].append(knn_precision)
    data_output['knn']['recall'].append(knn_recall)
    data_output['knn']['accuracy'].append(knn_acc)
    data_output['knn']['auc'].append(knn_roc_auc)
    data_output['knn']['fpr'].append(knn_fpr)
    data_output['knn']['tpr'].append(knn_tpr)
    
    data_output['dtree']['F1'].append(dtree_f1)
    data_output['dtree']['precision'].append(dtree_precision)
    data_output['dtree']['recall'].append(dtree_recall)
    data_output['dtree']['accuracy'].append(dtree_acc)
    data_output['dtree']['auc'].append(dtree_roc_auc)
    data_output['dtree']['fpr'].append(dtree_fpr)
    data_output['dtree']['tpr'].append(dtree_tpr)
    
    data_output['svm']['F1'].append(svm_f1)
    data_output['svm']['precision'].append(svm_precision)
    data_output['svm']['recall'].append(svm_recall)
    data_output['svm']['accuracy'].append(svm_acc)
    data_output['svm']['auc'].append(svm_roc_auc)
    data_output['svm']['fpr'].append(svm_fpr)
    data_output['svm']['tpr'].append(svm_tpr)
    
    
    
    #--------------deep learning method---------------------------------------
    lstm = LSTM()
    ann = ANN()
    cnn = CNN()
    
    if isGPU:
        lstm.cuda()
        ann.cuda()
        cnn.cuda()

    lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    ann_optimizer = torch.optim.Adam(ann.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
    # print the structure of the network
    print(lstm, ann, cnn)
    
    # data partition: 15% testing, 85% training
    training_data, test_data = torch.utils.data.random_split(all_data, [3344, 591])
    
    training_Loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_Loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    # training and testing
    lstm_test_loss_draw = []
    ann_test_loss_draw = []
    cnn_test_loss_draw = []
    
    lstm_loss_draw = []
    ann_loss_draw = []
    cnn_loss_draw = []
    
    for epoch in range(EPOCH):
        print('epoch {}'.format(epoch + 1))
        
        # training-----------------------------------------
        lstm_train_loss = 0.
        cnn_train_loss = 0.
        ann_train_loss = 0.
        
        lstm_train_acc = 0.
        cnn_train_acc = 0.
        ann_train_acc = 0.
        
        lstm.train()
        cnn.train()
        ann.train()
        
        
        
        for step, (batch_x, batch_y) in enumerate(training_Loader):
            batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
            batch_x_cnn = torch.unsqueeze(batch_x, dim=1).type(torch.float)
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_x_cnn = batch_x_cnn.cuda()
                batch_y = batch_y.cuda()
                
            output_cnn = cnn(batch_x_cnn)
            output_lstm = lstm(batch_x)
            output_ann = ann(batch_x)

            loss_cnn = loss_func(output_cnn, batch_y)
            loss_ann = loss_func(output_ann, batch_y)
            loss_lstm = loss_func(output_lstm, batch_y)
            
            lstm_train_loss += loss_lstm.item()
            cnn_train_loss += loss_cnn.item()
            ann_train_loss += loss_ann.item()
            
            if isGPU:
                lstm_pred = torch.max(output_lstm, 1)[1].cuda()
                cnn_pred = torch.max(output_cnn, 1)[1].cuda()
                ann_pred = torch.max(output_ann, 1)[1].cuda()
            else:
                lstm_pred = torch.max(output_lstm, 1)[1]
                cnn_pred = torch.max(output_cnn, 1)[1]
                ann_pred = torch.max(output_ann, 1)[1]
            
            lstm_train_correct = (lstm_pred == batch_y).sum()
            cnn_train_correct = (cnn_pred == batch_y).sum()
            ann_train_correct = (ann_pred == batch_y).sum()
            
            lstm_train_acc += lstm_train_correct.item()
            cnn_train_acc += cnn_train_correct.item()
            ann_train_acc += ann_train_correct.item()
            
            lstm_optimizer.zero_grad()
            cnn_optimizer.zero_grad()
            ann_optimizer.zero_grad()
            
            loss_lstm.backward()
            loss_cnn.backward()
            loss_ann.backward()
            
            lstm_optimizer.step()
            ann_optimizer.step()
            cnn_optimizer.step()
            
        print('LSTM:\n Train Loss: {:.6f}, Accuracy: {:.6f}\n'.format(lstm_train_loss / 
              (len(training_data)), lstm_train_acc / (len(training_data))))
        print('CNN:\n Train Loss: {:.6f}, Accuracy: {:.6f}\n'.format(cnn_train_loss / 
              (len(training_data)), cnn_train_acc / (len(training_data))))
        print('ANN:\n Train Loss: {:.6f}, Accuracy: {:.6f}\n'.format(ann_train_loss / 
              (len(training_data)), ann_train_acc / (len(training_data))))
        
        
        lstm_loss_draw.append(lstm_train_loss/(len(training_data)))
        cnn_loss_draw.append(cnn_train_loss/(len(training_data)))
        ann_loss_draw.append(ann_train_loss/(len(training_data)))
            
        
        # evaluation--------------------------------------------------
        lstm.eval()
        cnn.eval()
        ann.eval()
        
        lstm_eval_loss = 0.
        cnn_eval_loss = 0.
        ann_eval_loss = 0.
        
        lstm_eval_acc = 0.
        cnn_eval_acc = 0.
        ann_eval_acc = 0.
        
        lstm_final_prediction = np.array([])
        lstm_final_test = np.array([])
        lstm_f1_score = []
        lstm_recall = []
        lstm_precision = []
        
        cnn_final_prediction = np.array([])
        cnn_final_test = np.array([])
        cnn_f1_score = []
        cnn_recall = []
        cnn_precision = []
        
        ann_final_prediction = np.array([])
        ann_final_test = np.array([])
        ann_f1_score = []
        ann_recall = []
        ann_precision = []
        
        for step, (batch_x, batch_y) in enumerate(test_Loader):
            batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
            batch_x_cnn = torch.unsqueeze(batch_x, dim=1).type(torch.float)
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_x_cnn = batch_x_cnn.cuda()
                batch_y = batch_y.cuda()
    
            output_cnn = cnn(batch_x_cnn)
            output_lstm = lstm(batch_x)
            output_ann = ann(batch_x)

            
            loss_cnn = loss_func(output_cnn, batch_y)
            loss_ann = loss_func(output_ann, batch_y)
            loss_lstm = loss_func(output_lstm, batch_y)
            
            lstm_eval_loss += loss_lstm.item()
            cnn_eval_loss += loss_cnn.item()
            ann_eval_loss += loss_ann.item()
            
            lstm_pred = torch.max(output_lstm, 1)[1]
            cnn_pred = torch.max(output_cnn, 1)[1]
            ann_pred = torch.max(output_ann, 1)[1]
            
            lstm_train_correct = (lstm_pred == batch_y).sum()
            cnn_train_correct = (cnn_pred == batch_y).sum()
            ann_train_correct = (ann_pred == batch_y).sum()
            
            if isGPU:
                lstm_pred = torch.max(output_lstm, 1)[1].cuda()
                cnn_pred = torch.max(output_cnn, 1)[1].cuda()
                ann_pred = torch.max(output_ann, 1)[1].cuda()
            else:
                lstm_pred = torch.max(output_lstm, 1)[1]
                cnn_pred = torch.max(output_cnn, 1)[1]
                ann_pred = torch.max(output_ann, 1)[1]
            
            lstm_eval_acc += lstm_train_correct.item()
            cnn_eval_acc += cnn_train_correct.item()
            ann_eval_acc += ann_train_correct.item()
            
            # loss = loss_func(output, batch_y)
            # eval_loss += loss.item()
            # pred = torch.max(output, 1)[1]
            # num_correct = (pred == batch_y).sum()
            # eval_acc += num_correct.item()
            
            # F1 metrics
            lstm_final_prediction = np.concatenate((lstm_final_prediction, lstm_pred.cpu().numpy()), axis=0)
            lstm_final_test = np.concatenate((lstm_final_test, batch_y), axis=0)
            
            cnn_final_prediction = np.concatenate((cnn_final_prediction, cnn_pred.cpu().numpy()), axis=0)
            cnn_final_test = np.concatenate((cnn_final_test, batch_y), axis=0)
            
            ann_final_prediction = np.concatenate((ann_final_prediction, ann_pred.cpu().numpy()), axis=0)
            ann_final_test = np.concatenate((ann_final_test, batch_y), axis=0)
        
        lstm_f1_score.append(sklearn.metrics.f1_score(lstm_final_test, lstm_final_prediction, average='weighted').item())
        cnn_f1_score.append(sklearn.metrics.f1_score(cnn_final_test, cnn_final_prediction, average='weighted').item())
        ann_f1_score.append(sklearn.metrics.f1_score(ann_final_test, ann_final_prediction, average='weighted').item())

        lstm_recall.append(sklearn.metrics.recall_score(lstm_final_test, lstm_final_prediction, average='weighted').item())
        cnn_recall.append(sklearn.metrics.recall_score(cnn_final_test, cnn_final_prediction, average='weighted').item())
        ann_recall.append(sklearn.metrics.recall_score(ann_final_test, ann_final_prediction, average='weighted').item())

        lstm_precision.append(sklearn.metrics.precision_score(lstm_final_test, lstm_final_prediction, average='weighted').item())
        cnn_precision.append(sklearn.metrics.precision_score(cnn_final_test, cnn_final_prediction, average='weighted').item())
        ann_precision.append(sklearn.metrics.precision_score(ann_final_test, ann_final_prediction, average='weighted').item())
        
        print('LSTM:\n Test Loss: {:.6f}, Acc: {:.6f}'.format(lstm_eval_loss / 
              (len(test_data)), lstm_eval_acc / (len(test_data))))
        print('CNN:\n Test Loss: {:.6f}, Acc: {:.6f}'.format(cnn_eval_loss / 
              (len(test_data)), cnn_eval_acc / (len(test_data))))
        print('ANN:\n Test Loss: {:.6f}, Acc: {:.6f}'.format(ann_eval_loss / 
              (len(test_data)), ann_eval_acc / (len(test_data))))
        
        lstm_test_loss_draw.append(lstm_eval_loss/(len(test_data)))
        cnn_test_loss_draw.append(cnn_eval_loss/(len(test_data)))
        ann_test_loss_draw.append(ann_eval_loss/(len(test_data)))
        
        print('LSTM:\n F1: {}, recall: {}, precision: {}'.format(lstm_f1_score[-1], lstm_recall[-1], lstm_precision[-1]))
        print('CNN:\n F1: {}, recall: {}, precision: {}'.format(cnn_f1_score[-1], cnn_recall[-1], cnn_precision[-1]))
        print('ANN:\n F1: {}, recall: {}, precision: {}'.format(ann_f1_score[-1], ann_recall[-1], ann_precision[-1]))
        print('KNN:\n F1: {}, recall: {}, precision: {}'.format(knn_f1, knn_recall, knn_precision))
        print('SVM:\n F1: {}, recall: {}, precision: {}'.format(svm_f1, svm_recall, svm_precision))
        print('Dtree:\n F1: {}, recall: {}, precision: {}'.format(dtree_f1, dtree_recall, dtree_precision))
    
    
    # ROC curve and AUC
    lstm_test_y = label_binarize(lstm_final_test, classes=[0, 1, 2])
    lstm_pred_y = label_binarize(lstm_final_prediction, classes=[0, 1, 2])
    lstm_fpr, lstm_tpr, _ = roc_curve(lstm_test_y.ravel(), lstm_pred_y.ravel())
    lstm_roc_auc = auc(lstm_fpr, lstm_tpr)
    
    cnn_test_y = label_binarize(cnn_final_test, classes=[0, 1, 2])
    cnn_pred_y = label_binarize(cnn_final_prediction, classes=[0, 1, 2])
    cnn_fpr, cnn_tpr, _ = roc_curve(cnn_test_y.ravel(), cnn_pred_y.ravel())
    cnn_roc_auc = auc(cnn_fpr, cnn_tpr)
    
    ann_test_y = label_binarize(ann_final_test, classes=[0, 1, 2])
    ann_pred_y = label_binarize(ann_final_prediction, classes=[0, 1, 2])
    ann_fpr, ann_tpr, _ = roc_curve(ann_test_y.ravel(), ann_pred_y.ravel())
    ann_roc_auc = auc(ann_fpr, ann_tpr)
    
    
    data_output['lstm']['F1'].append(lstm_f1_score[-1])
    data_output['lstm']['precision'].append(lstm_precision[-1])
    data_output['lstm']['recall'].append(lstm_recall[-1])
    data_output['lstm']['accuracy'].append(lstm_eval_acc / (len(test_data)))
    data_output['lstm']['auc'].append(lstm_roc_auc)
    data_output['lstm']['fpr'].append(lstm_fpr)
    data_output['lstm']['tpr'].append(lstm_tpr)
    
    data_output['cnn']['F1'].append(cnn_f1_score[-1])
    data_output['cnn']['precision'].append(cnn_precision[-1])
    data_output['cnn']['recall'].append(cnn_recall[-1])
    data_output['cnn']['accuracy'].append(cnn_eval_acc / (len(test_data)))
    data_output['cnn']['auc'].append(cnn_roc_auc)
    data_output['cnn']['fpr'].append(cnn_fpr)
    data_output['cnn']['tpr'].append(cnn_tpr)
    
    data_output['ann']['F1'].append(ann_f1_score[-1])
    data_output['ann']['precision'].append(ann_precision[-1])
    data_output['ann']['recall'].append(ann_recall[-1])
    data_output['ann']['accuracy'].append(ann_eval_acc / (len(test_data)))
    data_output['ann']['auc'].append(ann_roc_auc)
    data_output['ann']['fpr'].append(ann_fpr)
    data_output['ann']['tpr'].append(ann_tpr)
    
    plt.figure()
    
    plt.plot(lstm_loss_draw, label='LSTM training')
    plt.plot(lstm_test_loss_draw, label='LSTM testing')
    
    plt.plot(ann_loss_draw, label='ANN training')
    plt.plot(ann_test_loss_draw, label='ANN testing')
    
    
    plt.plot(cnn_loss_draw, label='CNN training')
    plt.plot(cnn_test_loss_draw, label='CNN testing')
    
    plt.legend()
    
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, EPOCH+1, 1))
    plt.ylabel('Loss')
    plt.title('Loss Function')
    plt.savefig('../fig/'+'Loss_Kfold'+str(num_of_training)+'.png',dpi=500)
    
    # draw ROC curve in every epoch
    # plt.figure()
    # plt.plot(lstm_fpr, lstm_tpr, label='LSTM (AUC = {0:0.2f})'.format(lstm_roc_auc))
    # plt.plot(cnn_fpr, cnn_tpr, label='CNN (AUC = {0:0.2f})'.format(cnn_roc_auc))
    # plt.plot(ann_fpr, ann_tpr, label='ANN (AUC = {0:0.2f})'.format(ann_roc_auc))
    # plt.plot(knn_fpr, knn_tpr, label='KNN (AUC = {0:0.2f})'.format(knn_roc_auc))
    # plt.plot(svm_fpr, svm_tpr, label='SVM (AUC = {0:0.2f})'.format(svm_roc_auc))
    # plt.plot(dtree_fpr, dtree_tpr, label='DT (AUC = {0:0.2f})'.format(dtree_roc_auc))

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.legend()
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve')
    # plt.savefig('../fig/'+'ROC_Kfold'+str(num_of_training)+'.png',dpi=500)
    
pickle_out = open('dataoutput.pickle', 'wb')
pickle.dump(data_output, pickle_out)
pickle_out.close()
    
        