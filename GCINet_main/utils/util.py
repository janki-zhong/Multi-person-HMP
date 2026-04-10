import numpy as np
import torch
import torch.nn as nn
import math
import sys
from torch.nn import Parameter
# from utils.GCN import *
from scipy.spatial import ConvexHull
import torch.nn.functional as F
from sklearn.decomposition import PCA
# from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
def check_for_nan(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN values!")
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf values!")


"""      计算损失函数的不同方法      """
def mse(pred, target):
    mse_loss = (pred - target) ** 2
    mse_loss = mse_loss.sum(-1)
    loss = mse_loss.mean()
    return loss


def distance_loss(pred, target):
    # RMSE
    mse_loss = (pred - target) ** 2
    mse_loss = mse_loss.sum(-1)
    mse_loss = mse_loss.sqrt()
    loss = mse_loss.mean()
    return loss

# 将 PyTorch 张量转换为 NumPy 数组的函数
def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def parallel_dtw_loss(A_batch, B_batch):
    loss = 0
    for j in range(A_batch.shape[0]):  # 维度遍历
        seq_a = A_batch[j]
        seq_b = B_batch[j]
        distance, _ = fastdtw(seq_a, seq_b, dist=euclidean)
        loss += distance
    return loss

# 计算每对序列的DTW损失
def calculate_dtw_loss(A, B):
    total_loss = 0.0
    B_size, J_size, T_size, D_size = A.shape

    A_np = torch_to_numpy(A)
    B_np = torch_to_numpy(B)
    # 批次计算
    losses = Parallel(n_jobs=-1)(delayed(parallel_dtw_loss)(A_np[b], B_np[b]) for b in range(B_size))
    # 将losses转换为PyTorch张量并设置requires_grad=True
    mean_loss = torch.tensor(np.mean(losses), dtype=torch.float32, requires_grad=True)
    return mean_loss

def perceptron_loss(y_pred, y_true):
    # 感知损失
    # 计算每个样本的损失
    losses = torch.maximum(torch.tensor(0.0), -y_true * y_pred)
    # 求平均损失
    loss = torch.sum(losses) / len(y_true)
    return loss

def NRMSE(pred, target, oriented):
    nrmse = (pred - target) ** 2
    nrmse = nrmse.sum(-1)
    nrmse = nrmse.mean()
    nrmse = nrmse.sqrt()
    o = oriented.mean()
    loss = nrmse / o
    return loss


def RMSLE(pred, target):
    # 结果都是nan
    rmsle = torch.log(pred + 1) - torch.log(target + 1)
    rmsle = rmsle ** 2
    rmsle = rmsle.sum(-1).mean()
    loss = rmsle.sqrt()
    return loss


def log_cosh_loss(y_true, y_pred):
    # 计算误差
    error = y_true - y_pred
    # 计算 log(cosh(error))，即 log((e^error + e^(-error)) / 2)
    loss = torch.log(torch.cosh(error))
    # 返回平均损失
    return loss.mean()


def rrmse_loss(y_true, y_pred):
    # 计算均方根误差 (RMSE)
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    mean_true = torch.mean(y_true)
    rrmse = rmse / mean_true.abs()
    return rrmse

def logistic_loss(y_true, y_pred):
    # 出现的都是nan
    return -torch.sum(y_true*torch.log(y_pred)) + (1-y_true)*torch.log(1-y_pred)

def relation_loss(target, pred):
    mse_loss = torch.abs(pred - target)
    loss = mse_loss.mean()
    return loss


def getProbability(x):
    # x： 32,45,50,3
    B, NJ, T, D = x.shape
    x = x.permute(2, 0, 1, 3)  # T,B,NJ,D
    x = x[..., :1].sum(dim=-1).sum(dim=-1)
    # x = torch.log(x)
    x = F.softmax(x, dim=-1)  # 25,32
    return x


def BayesianProbability(y):
    y1 = y[:, :, :50]
    y2 = y[:, :, 50:]
    P1 = getProbability(y1)  # 50,32
    P2 = getProbability(y2)  # 25,32
    a = Parameter(torch.tensor(0.5))
    pca = PCA(n_components=25)
    P1 = pca.fit_transform(P1)
    P_bayes = a * P2 / P1
    return P_bayes


def kl_divergence_tensor(p, q):
    # 确保 p 和 q 是有效的概率分布
    p = p + 1e-10  # 防止为0
    q = q + 1e-10  # 防止为0
    # 计算 KL 散度 D_KL(P || Q)
    return torch.sum(p * torch.log(p / q))


class CosineSimilarityLoss(nn.Module):
    # 余弦相似度损失
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, output1, output2):
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(output1, output2)
        # 损失函数：1 - 余弦相似度
        loss = 1 - cosine_similarity.mean()
        return loss

def calculate_mse(X, Y):
    """
    计算均方误差 (Mean Squared Error)
    参数: X: 预测值，形状为 (B, NJ, T, D)  Y: 真实值，形状为 (B, NJ, T, D)
    返回:  MSE: 均方误差
    """
    # 确保输入的 X 和 Y 具有相同的形状
    assert X.shape == Y.shape, "输入数据和目标数据必须具有相同的形状"
    # 计算差值
    diff = X - Y
    # 计算平方
    squared_diff = diff ** 2
    # 计算均方误差
    mse = squared_diff.mean()  # 平均所有元素的平方误差
    return mse


def SetThreshold(pred, target):
    # 给GT划分范围，长度落到哪个区间，就属于多少的概率
    # (B,NJ,T,D)
    B, NJ, T, D = target.shape
    dis = (pred - target) ** 2  # 32，45，25，3
    dis = dis.sum(-1)  # 32，45，25
    dis = dis.sqrt()  # 32，45，25

    max_value = torch.max(dis)  # 13.1245
    min_value = torch.min(dis)  # 0.0709
    gap = max_value - min_value  # 13.0536
    cell = gap / 10  # 1.3054

    num = 0
    total = B * NJ * T
    for i in range(4):
        mask = (dis > cell * i) & (dis < cell * (i + 1))
        count = mask.sum().item()
        num = num + count
    probability = num / total
    return torch.tensor(probability, requires_grad=True)


def log_loss(p):
    loss = torch.log(p)
    loss = -loss
    return loss


"""     model中用到的其他方法      """
def process_pred1(pred_vel):  # , pred_vel_aux
    pred_vel_x = pred_vel[:, :, :50]
    pred_vel_y = pred_vel[:, :, 50:]
    return (pred_vel_x, pred_vel_y)

def process_pred(pred_vel):  # , pred_vel_aux
    pred_vel_x = pred_vel[:, :, :15]
    pred_vel_y = pred_vel[:, :, 15:]
    return (pred_vel_x, pred_vel_y)


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
    pred_vel_ = pred_vel.view(B, T, N, J, D).permute(0, 2, 3, 1, 4)  # B,N,J,T,D
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
    # exp_dis = torch.exp(-dis)
    return dis.cuda()


def vel_ab(vel_a, vel_b):
    relative_vel = vel_b - vel_a  # B,J,T,3
    # relative_vel = relative_vel.sum(-1)
    return relative_vel.cuda()


def acc_ab(pos_a, pos_b):
    # 计算两人之间的加速度信息（互动信息）   输入的是速度 B,J,T,3
    acc1 = torch.zeros(pos_a.shape)  # 加速度1
    acc1[:, :, 1:] = pos_a[:, :, 1:] - pos_a[:, :, :-1]
    acc2 = torch.zeros(pos_b.shape)  # 加速度2
    acc2[:, :, 1:] = pos_b[:, :, 1:] - pos_b[:, :, :-1]
    relative_acc = acc2 - acc1  # B,J,T,3
    # relative_acc = relative_acc.sum(-1)
    return relative_acc.cuda()


def distance_NJ(input_total):
    # 根据位置求距离    B,NJ,T,3   返回的维度为B,NJ,T
    pos = input_total[:, :, :, :3]  # (4,45,T,3)
    J = 15
    dis = []
    for i in range(0, pos.shape[1], J):
        for j in range(i + J, pos.shape[1], J):
            dis_ = distance_ab(pos[:, i:i + J], pos[:, j:j + J])  # B,J,T
            dis.append(dis_)
    dis = torch.cat(dis, dim=1)  # B,3J,T
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
                y.append(data[:, j * J:(j + 1) * J])
            else:
                continue
        y_ = torch.cat(y, dim=1)
        y_all.append(y_)
    return divide_data, y_all


""" 研二上改动   241014 """


def copy25(data):
    # B, NJ,50,9
    last_frame = data[:, :, -1, :]  # 形状为 (B, NJ, 9)
    # 复制最后一帧 25 次
    last_frame_repeated = last_frame.unsqueeze(2).repeat(1, 1, 25, 1)
    # 拼接到原始数据后面
    new_data = torch.cat((data, last_frame_repeated), dim=2)  # 新形状为 (B, NJ, T+25, 9)
    return new_data

def copy10(data):
    # B, NJ,50,9
    last_frame = data[:, :, -1, :]  # 形状为 (B, NJ, 9)
    # 复制最后一帧 25 次
    last_frame_repeated = last_frame.unsqueeze(2).repeat(1, 1, 10, 1)
    # 拼接到原始数据后面
    new_data = torch.cat((data, last_frame_repeated), dim=2)  # 新形状为 (B, NJ, T+25, 9)
    return new_data


def distanceMlp(input_total):
    # 根据位置求距离  # 32,45,76,3
    pos = input_total[:, :, :, :3]  # (4,45,60,3)
    pos_i = pos.unsqueeze(-3)  # (4,45,1,60,3)
    pos_j = pos.unsqueeze(-4)  # (4,1,45,60,3)
    pos_rel = pos_i - pos_j  # joint_pose的相对距离  (4,45,45,60,3)
    dis = torch.pow(pos_rel, 2).sum(-1)  # (32,45,45,75)
    dis = torch.sqrt(dis)
    # exp_dis = torch.exp(-dis)  # exp_dis：相对距离矩阵D_x (32,45,45,75)
    exp_dis = torch.mean(dis, dim=-2)
    return exp_dis


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


def mlpClassification(data):
    # data: B,NJ,75,9  100800
    data_ = data[..., :3].clone()  # B,NJ,T,D
    B, NJ, T, _ = data_.shape
    J = 15
    N = NJ // J
    # FC_dim = 1024
    dim = 128

    scint = nn.Sequential(
        # nn.Linear(B * J * T * _, FC_dim),
        # nn.Linear(FC_dim, dim),
        ######################################
        nn.Linear(B * J * T * _, dim),
        nn.ReLU(True),
        LN(dim),  # 3,256
        # nn.LayerNorm(dim),
        nn.Dropout(p=0.1),
        # nn.Flatten(),  # 1,3*256
        nn.Linear(dim, 2)).cuda()
    sftmx = nn.Softmax(dim=1).cuda()

    data_ = data_.reshape(B, N, J, T, _).reshape(N, -1)  # B,NJ,T,D ——> B,N,J,T,3  ——> N,-1
    # matrix = torch.zeros((data_.shape[1], data_.shape[1]))
    sct = scint(data_)
    probability = sftmx(sct)
    return probability


def mlpDis(data):
    # data: B,NJ,NJ,75  100800
    data_ = data.clone()  # B,NJ,T,D
    B, NJ, T = data_.shape
    J = 15
    N = NJ // J
    # FC_dim = 1024
    dim = 128

    scint = nn.Sequential(
        # nn.Linear(B * J * T * _, FC_dim),
        # nn.Linear(FC_dim, dim),

        nn.Linear(B * J * T, dim),
        nn.ReLU(True),
        # LN(dim),  # 3,256
        nn.LayerNorm(dim),
        nn.Dropout(p=0.1),
        # nn.Flatten(),  # 1,3*256
        nn.Linear(dim, 2)).cuda()
    sftmx = nn.Softmax(dim=1).cuda()

    data_ = data_.reshape(B, N, J, T).reshape(N, -1)  # B,NJ,T,D ——> B,N,J,T,3  ——> N,-1
    # matrix = torch.zeros((data_.shape[1], data_.shape[1]))
    sct = scint(data_)
    probability = sftmx(sct)
    return probability


def threshold_tensor(tensor):
    # 使用 torch 进行阈值操作，大于 0.5 的为 1，小于等于 0.5 的为 0
    return (tensor > 0.5).int()


def getSocial(data):
    # (3, 2)
    matrix = torch.zeros((data.shape[0], data.shape[0])).cuda()  # (3,3)
    # 计算每一列中 1 的数量
    count_of_ones = data.sum(dim=0)

    # 找到哪些列有两个 1
    columns_with_two_ones = (count_of_ones == 2).nonzero(as_tuple=True)[0]
    columns_with_three_ones = (count_of_ones > 2).nonzero(as_tuple=True)[0]  # 3个1

    if columns_with_three_ones.numel() > 0:
        return torch.zeros((data.shape[0], data.shape[0]))

    if columns_with_two_ones.numel() == 0:
        return torch.zeros((data.shape[0], data.shape[0]))

    # 提取出这些列的索引行
    # 需要检查每一行在这些列中是否有 1
    rows_indices = []
    for row_index in range(data.size(0)):
        if (data[row_index, columns_with_two_ones] == 1):
            rows_indices.append(row_index)

    # 转换为 Tensor
    rows_indices_tensor = torch.tensor(rows_indices).cuda()

    a = rows_indices_tensor[0]
    b = rows_indices_tensor[1]
    matrix[a, b] = 1
    matrix[b, a] = 1
    return matrix.cuda()


def remove_center(data):
    # data : B, NJ, 50, 9
    b, nj, t, _ = data.shape
    j = 15
    n = nj // j
    data = data.reshape(b, n, j, t, _)
    data = data - data[:, :, 0].unsqueeze(2)
    data = data.reshape(b, -1, t, _)
    return data




def getHead(data):
    # B, NJ, 75, 3  头 16-8， 颈部14-7
    b, nj, t, _ = data.shape
    j = 15
    n = nj // j
    data = data.reshape(b, n, j, t, _)
    head = (data[:, :, 8] - data[:, :, 7]).unsqueeze(2)  # 32,3,1,75,3
    head = head.repeat(1, 1, 15, 1, 1).reshape(b, -1, t, _)
    return head


class mlpHead(nn.Module):
    def __init__(self, input=75 * 9, dim=256, p=0.2, out=2):
        super().__init__()
        self.scint = nn.Sequential(
            nn.Linear(input, dim),
            nn.ReLU(True),
            # LN(dim),  # 3,256
            nn.LayerNorm(dim),
            nn.Dropout(p),
            nn.Linear(dim, out)
        ).cuda()

        self.sftmx = nn.Softmax(dim=1).cuda()

    def forward(self, head):
        # 32,45,75,3
        b, nj, t, _ = head.shape
        j = 15
        n = nj // j
        head = head.reshape(b, n, j, -1).reshape(n, -1)
        head = self.scint(head)
        head = self.sftmx(head)
        return head


def getInFrame(P1, P2):
    # P: B,J,T,3
    # 计算中点
    midpoints = (P1 + P2) / 2
    # 计算半径
    # 使用欧几里得距离公式计算每个点的半径
    diff = P1 - P2
    squared_diff = diff ** 2
    squared_distances = squared_diff.sum(dim=-1)  # 对 xyz 轴的差值平方求和
    radii = torch.sqrt(squared_distances)  # 计算欧几里得距离
    return midpoints, radii


def getSphere(data):
    # B, J, T, 3
    B, J, T, _ = data.shape

    # 获取中心点坐标
    center = data[:, 0]  # B,T,3
    center_expanded = center.unsqueeze(1).expand(-1, J, -1, -1)  # B,J,T,3

    # 计算距离，形状为 (B, J, T)
    distances = torch.norm(data - center_expanded, p=2, dim=-1)
    # 对每个样本 (B) 计算最大距离，返回最大距离和对应的索引
    max_distances, max_indices = distances.max(dim=1)  # 返回最大距离和相应的索引，shape: (B, T)

    return center, max_distances


# 计算欧几里得距离
def euclidean_distance(c1, c2):
    """计算两个球心之间的欧几里得距离"""
    return torch.norm(c1 - c2, dim=-1)


# 计算球体的体积
def sphere_volume(r):
    """计算球体的体积"""
    return (4 / 3) * np.pi * r ** 3


# 计算两个球体的交集体积
def sphere_intersection_volume(r1, r2, d):
    """计算两个球体的交集体积"""
    # 计算交集体积时，先判断交集是否存在
    mask_no_intersection = d >= r1 + r2  # 两个球体不相交
    mask_full_containment = d <= torch.abs(r1 - r2)  # 一个球体完全包含另一个球体

    # 初始化交集体积
    intersection_vol = torch.zeros_like(d)

    # 完全包含的情况
    intersection_vol[mask_full_containment] = (4 / 3) * np.pi * torch.min(r1, r2)[mask_full_containment] ** 3

    # 无交集的情况
    intersection_vol[mask_no_intersection] = 0.0

    # 计算部分交集的体积（根据交集公式）
    mask_partial_intersection = ~(mask_no_intersection | mask_full_containment)
    d_partial = d[mask_partial_intersection]
    r1_partial = r1[mask_partial_intersection]
    r2_partial = r2[mask_partial_intersection]

    term1 = (r1_partial + r2_partial - d_partial) ** 2
    term2 = d_partial ** 2 + 2 * d_partial * (r1_partial + r2_partial) - 3 * (r1_partial - r2_partial) ** 2
    intersection_vol[mask_partial_intersection] = (np.pi * term1 * term2) / (12 * d_partial)  # 32， 75

    return intersection_vol


# 计算交互比 (交集体积 / 并集体积)
def intersection_ratio(c1, c2, r1, r2):
    """计算两个球体的交互比 (交集体积 / 并集体积)"""
    # 计算球心之间的距离  B,T,3
    d = euclidean_distance(c1, c2)  # 32,75
    # 计算交集体积
    inter_volume = sphere_intersection_volume(r1, r2, d)  # 32, 75
    # 计算球体的体积
    vol1 = sphere_volume(r1)
    vol2 = sphere_volume(r2)
    # 计算并集体积
    union_volume = vol1 + vol2 - inter_volume
    # 计算交互比
    return inter_volume / union_volume


def compare(data, threshold_value):
    # IOU  32,75
    # 步骤1: 比较每个元素是否大于0
    greater_than_zero = data > threshold_value  # 返回一个32x75的布尔tensor
    # 步骤2: 对每一行，计算大于0的元素数量
    count_greater_than_zero = greater_than_zero.sum(dim=1)  # 结果为一个32维的tensor，每个元素表示每行大于0的数量
    # 步骤3: 判断每行是否满足大于0的元素数量 > 40
    # 创建一个可以训练的 threshold_value tensor
    # frame = torch.tensor(27, requires_grad=True).cuda()
    sample_is_1 = count_greater_than_zero > 27  # 结果为32维的布尔tensor，表示每行是否满足条件
    # 步骤4: 计算总共有多少个样本满足条件
    num_samples_is_1 = sample_is_1.sum().item()  # 计算满足条件的样本数量
    # 步骤5: 判断是否总共有16个或更多的样本满足条件
    result = 1 if num_samples_is_1 >= 16 else 0
    return result


def IoU(data):
    # B, NJ,75,9
    B, NJ, T, _ = data.shape
    j = 15
    n = NJ // j
    data = data.reshape(B, n, j, T, _)
    matrix = torch.zeros((n, n)).cuda()
    for i in range(n):
        for j in range(i + 1, n):
            # 1.得到交互框
            midpoints, radii = getInFrame(data[:, i, 0], data[:, j, 0])  # 32,75,2   32,75,(1)
            # 得到各自的球体
            center1, radi1 = getSphere(data[:, i])  # B,T,3; (B, T)
            center2, radi2 = getSphere(data[:, j])  # B,T,3; (B, T)
            # 求交互框和人之间的交互比
            # 使用广播机制一次性计算交互比
            ratios1 = intersection_ratio(midpoints, center1, radii, radi1)  # 32,75
            ratios2 = intersection_ratio(midpoints, center2, radii, radi2)  # 32,75
            # 求人与人之间的交互比
            # ratios3 = intersection_ratio(center1, center2, radi1, radi2)  # 32,75
            # 判断分组
            # 创建一个可以训练的 threshold_value tensor
            threshold_value = torch.tensor(0.3, requires_grad=True).cuda()
            # threshold_v2 = torch.tensor(0.3, requires_grad=True).cuda()
            # threshold_value = 0.9
            if compare(ratios1, threshold_value) & compare(ratios2, threshold_value):
                matrix[i, j] = 1
                matrix[j, i] = 1
            else:
                matrix[i, j] = 0
    return matrix
