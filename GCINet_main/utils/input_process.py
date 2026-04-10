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
            if index[i][j] == 1:
                y.append(divide_data[j])
            else:
                y.append(torch.zeros_like(divide_data[j]))
        y_ = torch.cat(y,dim=1)
        y_all.append(y_)
    return divide_data, y_all

def get_weight(data, index):
    #  (B,NJ,T,D)
    B, NJ, T, D = data.shape
    J = 15
    N = NJ // J
    data = data.reshape(B, N, J, T, D)

    divide_data = []
    for ii in range(N):
        data_ = data[:,ii]
        divide_data.append(data_)
    return divide_data

def matrix3to2(social, N):
# 将矩阵中，属于个人本身的部分去掉
    mask = ~torch.eye(N, dtype=torch.bool).cuda()
    non_diag_elements = social.masked_select(mask)
    matrix = non_diag_elements.reshape(N, N - 1)
    new_matrix = []
    for i in range(N):
        new_matrix.append(matrix[i])
    return new_matrix
