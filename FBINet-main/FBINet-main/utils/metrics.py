import numpy as np
import torch

def batch_VIM(GT, pred, select_frames=[1, 3, 7, 9, 13]):
    '''Calculate the VIM at selected timestamps.

    Args:
        GT: [B, T, J, 3].
    
    Returns:
        errorPose: [T].
    '''
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, axis=(2, 3))
    errorPose = np.sqrt(errorPose)
    errorPose = errorPose.sum(axis=0)
    return errorPose[select_frames]


def batch_MPJPE(GT, pred, select_frames=[1, 3, 7, 9, 13]):
    '''Calculate the MPJPE at selected timestamps.

    Args:
        GT: [B, T, J, 3], np.array, ground-truth pose position in world coordinate system (meter).
        GT: 地面实况姿态在世界坐标系中的位置（米）。
        pred: [B, T, J, 3], np.array, predicted pose position.

    Returns:
        errorPose: [T], MPJPE at selected timestamps.
    '''

    errorPose = np.power(GT - pred, 2)
    # B, T, J, 3
    errorPose = np.sum(errorPose, -1)
    errorPose = np.sqrt(errorPose)
    # B, T, J
    errorPose = errorPose.sum(axis=-1) / pred.shape[2]
    # B, T
    errorPose = errorPose.sum(axis=0)
    # T
    return errorPose[select_frames]

def batch_VIM_(GT, pred, select_frames=[1, 3, 7, 9, 13]):
    '''Calculate the VIM at selected timestamps.

    Args:
        GT: [B, J, T, 3].
    
    Returns:
        errorPose: [T].
    '''
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, axis=(1, 3))
    errorPose = np.sqrt(errorPose)
    errorPose = errorPose.sum(axis=0)
    return errorPose[select_frames]

def batch_MPJPE_(GT, pred, select_frames=[1, 3, 7, 9, 13]):
    '''Calculate the MPJPE at selected timestamps.

    Args:
        GT: [B, J, T, 3], np.array, ground-truth pose position in world coordinate system (meter).
        pred: [B, J, T, 3], np.array, predicted pose position.

    Returns:
        errorPose: [T], MPJPE at selected timestamps.
    '''

    errorPose = np.power(GT - pred, 2)
    # B, J, T, 3
    errorPose = np.sum(errorPose, -1)
    errorPose = np.sqrt(errorPose)
    # B, J, T
    errorPose = errorPose.sum(axis=1) / pred.shape[1]
    # B, T
    errorPose = errorPose.sum(axis=0)
    # T
    return errorPose[select_frames]


def JPE(V_pred, V_trgt, frame_idx):
    # 输入预测速度  (B,N,T,J,3) (1,3,45,15,3)
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :].cpu() - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2), dim=1).cpu().data.numpy().mean()
    return err * scale


def APE(V_pred, V_trgt, frame_idx):
    # 输入预测速度  (B,N,T,J,3)   (1,3,45,15,3)
    V_pred = (V_pred - V_pred[:, :, :, 0:1, :]).cpu()
    V_trgt = (V_trgt - V_trgt[:, :, :, 0:1, :]).cpu()
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :].cpu() - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()
    return err * scale

# def ADE(V_pred, V_trgt, frame_idx):
#     scale = 1000
#     err = np.arange(len(frame_idx), dtype=np.float_)
#     for idx in range(len(frame_idx)):
#         err[idx] = torch.linalg.norm(V_trgt[:, :, :frame_idx[idx], :1, :] - V_pred[:, :, :frame_idx[idx], :1, :], dim=-1).mean(1).mean()
#     return err * scale

def FDE(V_pred,V_trgt, frame_idx):
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.linalg.norm(V_trgt[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :].cpu() - V_pred[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :], dim=-1).mean(1).mean()
    return err * scale