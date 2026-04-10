import numpy as np
import torch
import math
import sys


def calculate_orientation_vector(rhip_, lhip_):
    # Calculate orientation data
    l = rhip_.clone()
    r = lhip_.clone()
    l[:, :, -1] = 0
    r[:, :, -1] = 0
    hip_line = l - r
    z_axis = torch.tensor([0.0, 0.0, 1.0])  # Calculate the positive z-axis vector
    vector = torch.cross(hip_line, z_axis.unsqueeze(0).unsqueeze(0).cuda())
    return vector


def is_parallel_or_opposite(v1, v2, tolerance=0.04):
    """
    Determine whether the directions of two vectors are parallel or relative (opposite directions)
    """
    #  B,T,3
    unit_vector1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    unit_vector2 = v2 / torch.norm(v2, dim=-1, keepdim=True)
    dot_product = torch.sum(unit_vector1 * unit_vector2, dim=2)  # Calculate dot product
    parallel = dot_product >= 1.0 - tolerance
    opposite = dot_product <= -1.0 + tolerance
    # Calculate quantity
    result = torch.logical_or(parallel, opposite)
    num_true = torch.sum(result)
    if num_true >= v1.shape[0] * 26:
        return 1
    else:
        return 0


def within_threshold(distance_tensor, threshold):
    return distance_tensor <= threshold


def calculate_similarity(coord1, coord2, threshold):
    euclidean_distance = torch.norm(coord1 - coord2, dim=2)
    within_thresh = within_threshold(euclidean_distance, threshold)
    num = torch.sum(within_thresh).item()
    if num >= coord1.shape[0] * 26:
        return 1
    else:
        return 0


def orientation_and_distance(data):
    # Preliminary identification of interactive interactions using orientation and distance  B,NJ,T,D
    data_ = data[..., :3].clone()  # B,NJ,T,D
    B, NJ, T, _ = data_.shape
    J = 15
    data_ = data_.reshape(B, -1, J, T, _)  # B,NJ,T,D ——> B,N,J,T,3
    data_[..., [0, 1, 2]] = data_[..., [2, 0, 1]]  # (Y,Z,X) --> (x,y,z)
    matrix = torch.zeros((data_.shape[1], data_.shape[1]))
    for j in range(data_.shape[1]):
        for i in range(j + 1, data_.shape[1]):
            # 1 Whether the orientation obtained from the hips/shoulders meets the conditions for obtaining results 1 and 2
            x1 = calculate_orientation_vector(data_[:, j, 4], data_[:, j, 1])  # hips
            x2 = calculate_orientation_vector(data_[:, i, 4], data_[:, i, 1])
            result1 = is_parallel_or_opposite(x1, x2, tolerance=0.3)  # return 0 or 1
            x11 = calculate_orientation_vector(data_[:, j, 12], data_[:, j, 9])  # shoulders
            x22 = calculate_orientation_vector(data_[:, i, 12], data_[:, i, 9])
            result2 = is_parallel_or_opposite(x11, x22, tolerance=0.3)
            # 2 Distance within a certain range
            trajectories_similar = calculate_similarity(data_[:, j, 0], data_[:, i, 0], threshold=1.5)  # return 0 or 1
            if (result1 or result2) and trajectories_similar:
                matrix[j, j + 1] = 1
                matrix[j + 1, j] = matrix[j, j + 1]
    return matrix.cuda()


# def Rotating_interaction(a,b):
#     # B,T,3  两人的身体交互点
#     bodyCenterPoint = (a - b) / 2.0  # B,T,3
#     r1 = a - bodyCenterPoint


def getheadori(head, neck):
    # 计算头部的朝向
    # B,T,3
    ori = head - neck
    return ori


# def isContans(v1, v2):
#     # v1 v2 B,T,3
#     # 计算点积
#     dot_product = np.sum(v1 * v2, axis=-1)  # 计算最后一维的点积，结果形状为 (32, 75)
#     # 计算向量的模
#     mag_a = np.linalg.norm(v1, axis=-1)
#     mag_b = np.linalg.norm(v2, axis=-1)
#     # 计算夹角的余弦值
#     cos_theta = dot_product / (mag_a * mag_b)
#     # 由于浮动误差，cos_theta 可能稍微超出 [-1, 1] 范围，因此需要进行裁剪
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     # 计算夹角（弧度制）
#     theta_rad = np.arccos(cos_theta)
#     # 将弧度转换为角度
#     angle = math.degrees(theta_rad)
#     count = 0
#     if 120 <= angle <= 180:
#         count += 1;


# def ori_and_dis_and_head(data):
#     # B, NJ,75,9
#     data_ = data[..., :3].clone()  # B,NJ,T,D
#     B, NJ, T, _ = data_.shape
#     J = 15
#     data_ = data_.reshape(B, -1, J, T, _)  # B,NJ,T,D ——> B,N,J,T,3
#     data_[..., [0, 1, 2]] = data_[..., [2, 0, 1]]  # (Y,Z,X) --> (x,y,z)
#     matrix = torch.zeros((data_.shape[1], data_.shape[1]))
#     for j in range(data_.shape[1]):
#         for i in range(j + 1, data_.shape[1]):
#             # 1 Whether the orientation obtained from the hips/shoulders meets the conditions for obtaining results 1 and 2
#             x1 = calculate_orientation_vector(data_[:, j, 4], data_[:, j, 1])  # hips
#             x2 = calculate_orientation_vector(data_[:, i, 4], data_[:, i, 1])
#             result1 = is_parallel_or_opposite(x1, x2, tolerance=0.3)  # return 0 or 1
#             x11 = calculate_orientation_vector(data_[:, j, 12], data_[:, j, 9])  # shoulders
#             x22 = calculate_orientation_vector(data_[:, i, 12], data_[:, i, 9])
#             result2 = is_parallel_or_opposite(x11, x22, tolerance=0.3)
#             # 2 Distance within a certain range
#             trajectories_similar = calculate_similarity(data_[:, j, 0], data_[:, i, 0], threshold=1.5)  # return 0 or 1
#             # 3 转圈互动
#
#             # 4 头部互动
#             head1 = getheadori(data_[:, i, 8], data_[:, i, 7])  # B,T,3
#             head2 = getheadori(data_[:, j, 8], data_[:, j, 7])
#             c3 = isContans(head1.cpu(), head2.cpu()).cuda()
#             # 5 手部互动
#             if (result1 or result2) and trajectories_similar:
#                 matrix[j, j + 1] = 1
#                 matrix[j + 1, j] = matrix[j, j + 1]
#     return matrix.cuda()


