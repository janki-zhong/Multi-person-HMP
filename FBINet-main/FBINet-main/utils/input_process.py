import torch
import torch.nn as nn
import numpy as np

def divide(data):
    # Split the data into individual data and return relevant personnel data based on the interaction situation (B,NJ,T,D)
    J = 15
    divide_data = []
    for ii in range(0,data.shape[1]-1,J):
        data_ = data[:,ii:ii+J]
        divide_data.append(data_)
    return divide_data

def Input_divide(data, index):
    # Split the data into individual data and return relevant personnel data based on the interaction situation  (B,NJ,T,D)  NxN
    J = 15
    divide_data = []
    y_all = []
    for ii in range(0,data.shape[1]-1, J):
        data_ = data[:,ii:ii+J]
        divide_data.append(data_)
    for i in range(len(divide_data)):
        y = []
        for j in range(len(divide_data)):
            if index[i,j] == 1:
                y.append(divide_data[j])
            else:
                y.append(torch.zeros_like(divide_data[j]))
        y_ = torch.cat(y,dim=1)
        y_all.append(y_)
    return divide_data, y_all
