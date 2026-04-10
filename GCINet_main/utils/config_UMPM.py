import argparse
import os

## data
dataset_skip = 2
strike = 2

## train & test
learning_rate = 1e-3
num_epoch = 100
step_size = 10
gamma = 0.8

cuda_devices = '0'
pretrain_path = ''

weight_loss_pred = 50.0
weight_loss_recon = 10.0

## model setting
depth = 9
num_heads = 16

model_path = 'logs/d_9_nh_16_04101852/best.pt'  #  Our  这个模型应该不匹配了，自己重新复现生成吧
"""我测试了train的相关文件，可以运行
    初始时，在短期预测结果中，我的depth=6，效果最佳
    但在做depth的消融实验时，我额外考虑了长期预测的效果，权衡之下选择了depth=9
    你可以按照你的想法进行更改
"""


def parse_args():
    parser = argparse.ArgumentParser()

    ## path setting
    parser.add_argument('--log_dir',
                        type=str,
                        default=os.path.join(os.getcwd(), 'logs/'),
                        help='dir for saving logs')

    parser.add_argument('--log2_dir',
                        type=str,
                        default=os.path.join(os.getcwd(), 'logs2/'),
                        help='dir for saving mocap logs')

    ## train & test setting
    parser.add_argument('--lr',
                        type=float,
                        default=learning_rate,
                        help='initial learing rate')
    parser.add_argument('--num_epoch',
                        type=int,
                        default=num_epoch,
                        help='#epochs to train')
    parser.add_argument('--step_size',
                        type=int,
                        default=step_size,
                        help='learning rate decay step')
    parser.add_argument('--gamma',
                        type=float,
                        default=gamma,
                        help='learning rate decay rate')

    parser.add_argument('--device',
                        type=str,
                        default=cuda_devices,
                        help='set device for training')
    parser.add_argument('--pretrain_path',
                        type=str,
                        default=pretrain_path,
                        help='path of the model pretrained on AMASS')
    parser.add_argument('--model_path',
                        type=str,
                        default=model_path,
                        help='path of the model finetuned on 3dpw')

    parser.add_argument('--weight_loss_pred',
                        type=float,
                        default=weight_loss_pred,
                        help='loss weight of predicted pose loss')
    parser.add_argument('--weight_loss_recon',
                        type=float,
                        default=weight_loss_recon,
                        help='loss weight of reconstruction pose loss')

    ## model setting
    parser.add_argument('--depth',
                        type=int,
                        default=depth,
                        help='model depth')
    parser.add_argument('--num_heads',
                        type=int,
                        default=num_heads,
                        help='num heads of multihead attention')
    ## data process setting
    parser.add_argument('--skip',
                        type=int,
                        default=dataset_skip,
                        help='down sample rate')
    parser.add_argument('--strike',
                        type=int,
                        default=strike,
                        help='number of frames that we have to skip')
    parser.add_argument('--rc',
                        type=bool,
                        default=True,
                        help='whether to remove camera movement')

    args = parser.parse_args()

    return args
