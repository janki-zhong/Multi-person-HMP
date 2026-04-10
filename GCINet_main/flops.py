from time import time

import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from utils.util import *
from models.model import JRTransformer
from utils.input_process import *

"""Griffin"""
T1 = 50 + 25  # + 25 Input
T2 = 50 + 25  # Output
F = 6  # The last dimension
num_heads = 16
depth = 6

model = JRTransformer(in_joint_size=T1 * F,
                                   feat_size=512,
                                   out_joint_size=T2 * 3,
                                   num_heads=num_heads, depth=depth).cuda()

input = torch.randn(32, 45, 75, 9)  # 假设输入尺寸为1x1x28x28
social = IoU(input[..., :3].cuda())

flops, params = profile(model, inputs=(input[..., 3:], social))
# flops, params = clever_format([flops, params], "%.3f")

print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）

""" 计算时间 """
# # 使用 torch.cuda.Event 进行 GPU 时间测量
# torch.cuda.synchronize()
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
#
# start.record()
# result = model(input[..., 3:], social)
# end.record()
#
# torch.cuda.synchronize()  # 等待事件完成
# elapsed_time = start.elapsed_time(end) / 1000  # 转换为秒
# print(f"Inference time: {elapsed_time} seconds")

"""FBINet"""
# T1 = 50 + 25  # + 25 Input
# T2 = 50 + 25  # Output
# F = 6  # The last dimension
# num_heads = 16
# depth = 6
#
# model = JRTransformer(in_joint_size=T1 * F,
#                                    feat_size=512,
#                                    out_joint_size=T2 * 3,
#                                    num_heads=num_heads, depth=depth).cuda()
#
# input = torch.randn(32, 45, 75, 9)  # 假设输入尺寸为1x1x28x28
# social = IoU(input[..., :3].cuda())
# x_input_joint, y_input_joint = Input_divide(input[..., 3:], social)
#
# flops = 0
# params = 0
# for m in range(len(x_input_joint)):
#     f, p = profile(model, inputs=(x_input_joint[m], y_input_joint[m])) #* len(x_input_joint)
#     flops = flops + f
#     params = params + p
#
# print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
# print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）