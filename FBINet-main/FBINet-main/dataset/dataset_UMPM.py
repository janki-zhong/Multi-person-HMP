import torch.utils.data as data
import torch
import numpy as np
import copy
import open3d as o3d


class Data(data.Dataset):
    def __init__(self, dataset, mode=0, device='cuda', transform=False, opt=None):
        if dataset == "mocap_umpm":
            if mode == 0:
                self.data = np.load('data/Mocap_UMPM/train_3_75_mocap_umpm.npy')   # (13000,3,75,45)
                #self.data = self.data[:32]
            else:
                self.data = np.load('data/Mocap_UMPM/test_3_75_mocap_umpm.npy')  # (3000,3,75,45)
        if dataset == "mupots":  # two modes both for evaluation
            if mode == 0:
                self.data = np.load('data/MuPoTs3D/mupots_150_2persons.npy')[:, :, ::2, :] # (176,2,75,45)
            if mode == 1:
                self.data = np.load('data/MuPoTs3D/mupots_150_3persons.npy')[:, :, ::2, :] # (192,3,75,45)
        if dataset == "mix1":
            if mode == 1:
                self.data = np.load('data/mix1_6persons.npy')  # (1000,6,75,45)
        if dataset == "mix2":
            if mode == 1:
                self.data = np.load('data/mix2_10persons.npy')   # (1000,10,75,45)

        self.len = len(self.data)
        self.device = device
        self.dataset = dataset
        self.transform = transform
        self.input_time = 50

    def __getitem__(self, index):
        data = self.data[index]

        if self.transform:  # radomly rotate the scene for augmentation
            idx = np.random.randint(0, 3)
            rot = [np.pi, np.pi / 2, np.pi / 4, np.pi * 2]
            points = self.data[index].reshape(-1, 3)
            # Read point
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # Point rotation
            pcd_EulerAngle = copy.deepcopy(pcd)
            R1 = pcd.get_rotation_matrix_from_xyz((0, rot[idx], 0))
            pcd_EulerAngle.rotate(R1)
            pcd_EulerAngle.paint_uniform_color([0, 0, 1])
            data = np.asarray(pcd_EulerAngle.points).reshape(-1, 75, 45)

        input_seq = data[:, :self.input_time, ]
        output_seq = data[:, self.input_time:, :]

        input_seq = torch.as_tensor(input_seq, dtype=torch.float32).cuda()
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32).cuda()
        last_input = input_seq[:, -1:, :]
        output_seq = torch.cat([last_input, output_seq], dim=1)
        input_seq = input_seq.reshape(input_seq.shape[0], input_seq.shape[1], -1)
        output_seq = output_seq.reshape(output_seq.shape[0], output_seq.shape[1], -1)

        # input
        # Calculate velocity based on location
        J = 15
        input_seq = input_seq.reshape(input_seq.shape[0], input_seq.shape[1], J, -1)
        vel_data = torch.zeros(input_seq.shape).cuda()
        vel_data[:, 1:, :, :] = input_seq[:, 1:] - input_seq[:, :-1]
        # Calculate instantaneous directional information
        len_data = torch.zeros(input_seq.shape).cuda()  # N,T,J,3
        len_data[:, :, 1:, :] = input_seq[:, :, 1:, :] - input_seq[:, :, :-1, :]
        # concat positon, velocity, instantaneous direction
        input_data = torch.cat((input_seq, vel_data, len_data), dim=-1)
        input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], -1)

        # output
        J = 15
        output_seq = output_seq.reshape(output_seq.shape[0], output_seq.shape[1], J, -1)
        vel_data = torch.zeros(output_seq.shape)
        vel_data[:, 1:, :, :] = output_seq[:, 1:] - output_seq[:, :-1]  # N,T,J,3
        # concat positon, velocity
        output_data = torch.cat((output_seq, vel_data.cuda(), ), dim=-1)
        output_data = output_data.reshape(output_data.shape[0], output_data.shape[1], -1)

        return input_data, output_data

    def __len__(self):
        return self.len

