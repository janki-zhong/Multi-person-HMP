import numpy as np
import torch
import math
import sys
from torch.nn import Parameter

def distance_loss(target, pred):
    mse_loss = (pred - target) ** 2
    mse_loss = mse_loss.sum(-1)
    mse_loss = mse_loss.sqrt()
    loss = mse_loss.mean()
    return loss

def relation_loss(target, pred):
    mse_loss = torch.abs(pred - target) 
    loss = mse_loss.mean()
    return loss

def process_pred(pred_vel, pred_vel_aux):
	pred_vel_x = pred_vel[:, :, :50]
	pred_vel_y = pred_vel[:, :, 50:]
	pred_vel_aux_x = []
	pred_vel_aux_y = []
	for pred_ in pred_vel_aux:
		pred_vel_aux_x.append(pred_[:, :, :50])
		pred_vel_aux_y.append(pred_[:, :, 50:])
	return pred_vel_x, pred_vel_y, pred_vel_aux_x, pred_vel_aux_y

def get_length(pred_vel):
	# 得到关节之间的长度信息
	# 得到长度信息
	B, T, NJ, D = pred_vel.shape
	N = 3
	J = 15
	pred_vel_ = pred_vel.view(B, T, N, J, D)
	pred_len = torch.zeros((B, T, N, J, D))
	pred_len[:, :, :, 1:, :] = pred_vel_[:, :, :, 1:, :] - pred_vel_[:, :, :, :-1, :]
	pred_len = pred_len.view(B, T, NJ, D).cuda()
	return pred_len

def get_acceleration(pred_vel):
	# 得到关节之间的加速度信息
	B, T, NJ, D = pred_vel.shape
	N = 3
	J = 15
	pred_vel_ = pred_vel.view(B, T, N, J, D).permute(0,2,3,1,4)  # B,N,J,T,D
	pred_acc = torch.zeros(pred_vel_.shape)
	pred_acc[:, :, :, 1:, :] = pred_vel_[:, :, :, 1:, :] - pred_vel_[:, :, :, :-1, :]
	pred_acc = pred_acc.permute(0, 3, 1, 2, 4)
	pred_acc = pred_acc.reshape(B, T, NJ, D).cuda()
	return pred_acc

def get_distance(pred_pos):
	# 计算两点间的距离
	B, T, NJ, D = pred_pos.shape
	dis_data = torch.zeros((B, T, NJ, D))
	dis_data = torch.pow(pred_pos, 2).sum(-1)
	dis = torch.sqrt(dis_data).unsqueeze(3).cuda()
	return dis

def get_position(input_data, pred_vel):
	# 由速度得到位置信息
	last_motion = input_data[:, :, :, :3].permute(0, 2, 1, 3)  # 区别在于input_ori没经过相机删除
	pred_motion = (pred_vel.cumsum(dim=1) + last_motion[:, -1:])  # 利用速度得到15帧位置关系（B，T=15，NJ，3）
	return pred_motion

def cat(pred_motion, pred_vel):
	# 将位置、速度信息concat在一起
	input_joint = torch.cat((pred_motion, pred_vel), dim=-1)  # (B,T,NJ,D=6)
	input_joint = input_joint.permute(0, 2, 1, 3)  # (B,NJ,T,D)
	input_joint = input_joint.float().cuda()  # 转换到cuda上计算，否则无法代入模型
	return input_joint

def distance(input_total):
	# 根据位置求距离  # 32,45,76,3
	pos = input_total[:, :, :, :3]  # (4,45,60,3)
	pos_i = pos.unsqueeze(-3)  # (4,45,1,60,3)
	pos_j = pos.unsqueeze(-4)  # (4,1,45,60,3)
	pos_rel = pos_i - pos_j  # joint_pose的相对距离  (4,45,45,60,3)
	dis = torch.pow(pos_rel, 2).sum(-1)  # (4,45,45,60)
	dis = torch.sqrt(dis)
	exp_dis = torch.exp(-dis)  # exp_dis：相对距离矩阵D_x (4,45,45,60)
	return exp_dis

def distance_ab(pos_a, pos_b):
	# 计算距离（用于鉴别器） B,J,T,3  适用于model_多人
	dis = pos_b - pos_a  # B,15,15,3
	dis = torch.pow(dis, 2).sum(-1)  # B,15,50
	dis = torch.sqrt(dis)  # B,15,T=50
	#exp_dis = torch.exp(-dis)
	return dis.cuda()

def vel_ab(vel_a, vel_b):
	relative_vel = vel_b - vel_a # B,J,T,3
	#relative_vel = relative_vel.sum(-1)
	return relative_vel.cuda()

def acc_ab(pos_a, pos_b):
	# 计算两人之间的加速度信息（互动信息）   输入的是速度 B,J,T,3
	acc1 = torch.zeros(pos_a.shape)  # 加速度1
	acc1[:, :, 1:] = pos_a[:, :, 1:] - pos_a[:, :, :-1]
	acc2 = torch.zeros(pos_b.shape)  # 加速度2
	acc2[:, :, 1:] = pos_b[:, :, 1:] - pos_b[:, :, :-1]
	relative_acc = acc2 - acc1  # B,J,T,3
	#relative_acc = relative_acc.sum(-1)
	return relative_acc.cuda()

def distance_NJ(input_total):
	# 根据位置求距离    B,NJ,T,3   返回的维度为B,NJ,T
	pos = input_total[:, :, :, :3]  # (4,45,T,3)
	J = 15
	dis = []
	for i in range(0, pos.shape[1], J):
		for j in range(i+J, pos.shape[1], J):
			dis_ = distance_ab(pos[:, i:i+J], pos[:, j:j+J])  # B,J,T
			dis.append(dis_)
	dis = torch.cat(dis,dim=1)  # B,3J,T
	return dis

def divide_2(data):
	# B,NJ,T,9
	J = 15
	N = data.shape[1] // J
	divide_data = []
	y_all = []
	for ii in range(0, data.shape[1] - 1, J):
		i = ii // J
		data_ = data[:, ii:ii + J]
		divide_data.append(data_)

		index = torch.ones(N)
		index[i] = 0
		y = []
		for j in range(N):
			if index[j] == 1:
				y.append(data[:,j*J:(j+1)*J])
			else:
				continue
		y_ = torch.cat(y, dim=1)
		y_all.append(y_)
	return divide_data, y_all









