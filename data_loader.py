#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:40:52 2020

@author: aran-lq
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class PMUdataset(Dataset):
    
    def __init__(self, path):
        
        #Where is the raw data
        normal_path = path + 'normal/'
        fault_1_path = path + 'DCDC_fault/'
        fault_2_path = path + 'VSC_delay/'
        
        normal_files = [f for f in os.listdir(normal_path) if f.endswith('.csv')]
        fault_1_files = [f for f in os.listdir(fault_1_path) if f.endswith('.csv')]
        fault_2_files = [f for f in os.listdir(fault_2_path) if f.endswith('.csv')]
        
        np_normal = []
        np_fault_1 = []
        np_fault_2 = []
        
        for f in normal_files:
            temp_file = np.loadtxt(normal_path + f, delimiter=',', dtype=np.float32)
            np_normal.append(temp_file)
        
        for f in fault_1_files:
            temp_file = np.loadtxt(fault_1_path + f, delimiter=',', dtype=np.float32)
            np_fault_1.append(temp_file)
        
        for f in fault_2_files:
            temp_file = np.loadtxt(fault_2_path + f, delimiter=',', dtype=np.float32)
            np_fault_2.append(temp_file)
        
        # x: input data, y: target
        self.y_data = np.concatenate([np.zeros(len(np_normal),dtype=np.long), np.zeros(len(np_fault_1),dtype=np.long)+1, np.zeros(len(np_fault_2), dtype=np.long)+2, ])
        self.x_data = np.concatenate([np_normal, np_fault_1, np_fault_2])
        self.len = self.x_data.shape[0]
        # self.x_data = torch.from_numpy(self.x_data)
        # self.y_data = torch.from_numpy(self.y_data)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

