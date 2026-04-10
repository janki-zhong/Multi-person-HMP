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

        self.T1 = 50  # Input
        self.T2 = 50+25  # Output
        self.F = 9  # The last dimension

        self.model = JRTransformer(in_joint_size=self.T1 * self.F,
                                   feat_size=512,
                                   out_joint_size=self.T2 * 3,
                                   num_heads=args.num_heads, depth=args.depth).to(self.device)

        self.rc = args.rc


        test_dataset = Data(dataset='mocap_umpm', mode=1, device=args.device, transform=False)
        print("Load Test set!")
        self.test_dataloader = DataLoader(test_dataset, batch_size=30, shuffle=False, drop_last=True)

        self.edges = np.array(
            [[0, 1], [1, 2], [2, 3], [0, 4],[4, 5], [5, 6], [0, 7],
             [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
        )

        # The trained model
        self.path = args.model_path

    def test(self):
        path = self.path
        # Load the trained model
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()

        frame_idx = [5, 10, 15, 20, 25]
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
                output = output_seq.reshape(B, N, 26, J, -1).permute(0, 1, 3, 2, 4).reshape(B, -1, 26, 6)
                input_data_original = input_data_original.float().cuda()
                input_total = input_data_original.clone()
                input_data_original = input_data_original.permute(0, 2, 1, 3, 4)
                input_total = input_total.permute(0, 2, 1, 3, 4)

                batch_size = input_total.shape[0]

                if self.rc:
                    camera_vel = input_total[:, 1:40, :, :, 3:6].mean(dim=(1, 2, 3))  # B, 3
                    input_total[..., 3:6] -= camera_vel[:, None, None, None]
                    input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:6].cumsum(dim=1)

                input_data_original = input_data_original.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 50, 9)
                input_joint = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 50, 9)

                """ Interaction Perceptron """
                social = orientation_and_distance(input_joint)

                # Processing input data: Split the input_joint of three people into individual x_input_joint.
                x_input_joint, y_input_joint = Input_divide(input_joint, social)

                # Input individually
                pred_motion = []
                input_data_original = divide(input_data_original)
                for m in range(len(x_input_joint)):
                    pred_vel_x, pred_vel_aux_x = self.model(x_input_joint[m], y_input_joint[m])
                    # B,J,T,3
                    pred_vel = pred_vel_x[:, :, 50:]
                    pred_vel = pred_vel.permute(0, 2, 1, 3)  # B，T，J，3
                    if self.rc:
                        pred_vel = pred_vel + camera_vel[:, None, None]  # B, T, J, 3
                    pred_motion_ = get_position(input_data_original[m], pred_vel)
                    pred_motion_ = pred_motion_.permute(0, 2, 1, 3)  # B，J，T，3
                    pred_motion.append(pred_motion_)
                pred_motion = torch.cat(pred_motion, dim=1)


                """calculate metrics：JPE、APE、FDE"""
                B, NJ, T, D = pred_motion.shape
                gt_v = output[..., :3].reshape(B, -1, 15, T+1, 3).permute(0, 1, 3, 2, 4).cpu()  # B,N,T,J,3
                # (B,NJ,T,3) --> B,N,T,J,3
                pred_v = pred_motion.reshape(B, -1, 15, T, D).permute(0, 1, 3, 2, 4).cpu()  # B,N,T,J,3

                n += 1
                jpe_err = JPE(pred_v, gt_v, frame_idx)
                ape_err = APE(pred_v, gt_v, frame_idx)
                fde_err = FDE(pred_v, gt_v, frame_idx)

                ape_err_total += ape_err
                jpe_err_total += jpe_err
                fde_err_total += fde_err

            print("{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d} | {5:6d}| {6: <16}".format("Lengths", 200, 400, 600, 800, 1000, "average"))
            print("=== JPE Test Error ===")
            jpe_average = (jpe_err_total[0]/n + jpe_err_total[2]/n +jpe_err_total[4]/n)/3
            print(
                "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f} | {6:6.0f}".format("Our", jpe_err_total[0] / n,
                                                                                         jpe_err_total[1] / n,
                                                                                         jpe_err_total[2] / n,
                                                                                         jpe_err_total[3] / n,
                                                                                         jpe_err_total[4] / n, jpe_average))
            print("=== APE Test Error ===")
            ape_average = (ape_err_total[0] / n + ape_err_total[2] / n + ape_err_total[4] / n) / 3
            print(
                "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f} | {6:6.0f}".format("Our", ape_err_total[0] / n,
                                                                                         ape_err_total[1] / n,
                                                                                         ape_err_total[2] / n,
                                                                                         ape_err_total[3] / n,
                                                                                         ape_err_total[4] / n, ape_average))
            print("=== FDE Test Error ===")
            fde_average = (fde_err_total[0] / n + fde_err_total[2] / n + fde_err_total[4] / n) / 3
            print(
                "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f} | {6:6.0f}".format("Our", fde_err_total[0] / n,
                                                                                         fde_err_total[1] / n,
                                                                                         fde_err_total[2] / n,
                                                                                         fde_err_total[3] / n,
                                                                                         fde_err_total[4] / n, fde_average))
            print("{1:6d}".format("n", n))


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tester = Tester(args)
    tester.test()
