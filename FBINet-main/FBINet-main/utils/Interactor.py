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
    if num_true >= v1.shape[0]* 26:
        return 1
    else:
        return 0


def within_threshold(distance_tensor, threshold):
    return distance_tensor <= threshold

def calculate_similarity(coord1, coord2, threshold):
    euclidean_distance = torch.norm(coord1 - coord2, dim=2)
    within_thresh = within_threshold(euclidean_distance, threshold)
    num = torch.sum(within_thresh).item()
    if num >= coord1.shape[0]*26:
        return 1
    else:
        return 0

def orientation_and_distance(data):
    # Preliminary identification of interactive interactions using orientation and distance  B,NJ,T,D
    data_ = data[..., :3].clone()  # B,NJ,T,D
    B,NJ,T,_ = data_.shape
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
