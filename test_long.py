import sys

sys.path.append(".")
import os
import numpy as np
import torch
from models.model import JRTransformer
from dataset.dataset_UMPM import Data
from torch.utils.data import DataLoader
from utils.config_UMPM import *
from utils.util import *
from utils.metrics import FDE, JPE, APE
from utils.Interactor import *
from utils.input_process import *


class Tester:
    def __init__(self, args):
        # Set cuda device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(0)
        else:
            self.device = torch.device("cpu")
        print('Using device:', self.device)
        self.cuda_devices = args.device

        self.T1 = 15   # Input
        self.T2 = 60  # Output
        self.F = 6  # The last dimension

        self.model = JRTransformer(in_joint_size=self.T1 * self.F,
                                   feat_size=512,
                                   out_joint_size=self.T2 * 3,
                                   num_heads=args.num_heads, depth=args.depth).to(self.device)

        self.rc = args.rc

        test_dataset = Data(dataset='mix1', mode=1, device=args.device, transform=False)  # mocap_umpm mupots
        print("Load Test set!")
        self.test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=True)

        # The trained model
        self.path = args.model_path

    def test(self):
        path = self.path
        # Load the trained model
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()

        frame_idx = [15, 30, 45]
        n = 0
        ape_err_total = np.arange(len(frame_idx), dtype=np.float_)
        jpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
        fde_err_total = np.arange(len(frame_idx), dtype=np.float_)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader, 0):
                input_data_original, output_seq = data
                B, N, T, D = input_data_original.shape
                J = 15

                input_data_original = input_data_original.reshape(B, N, T, J, -1)  # B,N,T,J,9
                output = output_seq.reshape(B, N, 46, J, -1).permute(0, 1, 3, 2, 4).reshape(B, -1, 46, 6)
                input_data_original = input_data_original.float().cuda()
                input_total = input_data_original.clone()
                input_data_original = input_data_original.permute(0, 2, 1, 3, 4)
                input_total = input_total.permute(0, 2, 1, 3, 4)

                batch_size = input_total.shape[0]

                if self.rc:
                    camera_vel = input_total[:, 1:15, :, :, 3:6].mean(dim=(1, 2, 3))  # B, 3
                    input_total[..., 3:6] -= camera_vel[:, None, None, None]
                    input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:6].cumsum(dim=1)

                input_data_original = input_data_original.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 15, 9)
                input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 15, 9)

                """ Interaction Perceptron """
                # 利用交互框，算交并比来进行交互分组
                social = IoU(input_total[..., :3])

                #input_joint = copy25(input_total)
                input_joint = input_total

                """无交互的人相当于信息权重为0~0.3"""
                pred_vel_x = self.model(input_joint[..., 3:], social)
                pred_vel = pred_vel_x[:, :, 15:]
                pred_vel = pred_vel.permute(0, 2, 1, 3)
                if self.rc:
                    pred_vel = pred_vel + camera_vel[:, None, None]
                pred_motion = get_position(input_data_original, pred_vel)
                pred_motion = pred_motion.permute(0, 2, 1, 3)  # B，NJ，T，3

                """calculate metrics：JPE、APE、FDE"""
                B, NJ, T, D = pred_motion.shape
                gt_v = output[..., :3].reshape(B, -1, 15, T + 1, 3).permute(0, 1, 3, 2, 4).cpu()  # B,N,T,J,3
                # (B,NJ,T,3) --> B,N,T,J,3
                pred_v = pred_motion.reshape(B, -1, 15, T, D).permute(0, 1, 3, 2, 4).cpu()  # B,N,T,J,3

                np.save('train_lxy2_pred_1028.npy', pred_v)

                n += 1
                jpe_err = JPE(pred_v, gt_v, frame_idx)
                ape_err = APE(pred_v, gt_v, frame_idx)
                fde_err = FDE(pred_v, gt_v, frame_idx)

                ape_err_total += ape_err
                jpe_err_total += jpe_err
                fde_err_total += fde_err

            print(
                "{0: <16} | {1:6d} | {2:6d} | {3:<16}".format("Lengths", 1, 2, 3, "average"))
            print("=== JPE Test Error ===")
            jpe_average = (jpe_err_total[0] / n + jpe_err_total[1] / n + jpe_err_total[2] / n) / 3
            print(
                "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f}".format("Our",jpe_err_total[0] / n,
                                                                                          jpe_err_total[1] / n,
                                                                                          jpe_err_total[2] / n,
                                                                                          jpe_average))
            print("{1:6d}".format("n", n))


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tester = Tester(args)
    tester.test()
