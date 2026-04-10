import sys
sys.path.append(".")
import os
import random
import numpy as np
import torch
import time
from models.model import JRTransformer
from dataset.dataset_UMPM import Data
from torch.utils.data import DataLoader
from utils.config_UMPM import *
from utils.util import *
from datetime import datetime
from utils.metrics import FDE, JPE, APE
from utils.Interactor import *
from utils.input_process import *

class Trainer:
    def __init__(self, args):
        # Set cuda device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(0)
        else:
            self.device = torch.device("cpu")
        print('Using device:', self.device)
        self.cuda_devices = args.device

        # Training parameters
        self.learning_rate = args.lr
        self.num_epoch = 100
        self.weight_loss_pred = args.weight_loss_pred
        self.weight_loss_recon = args.weight_loss_recon
        self.weight_loss_aux = args.weight_loss_aux
        self.T1 = 50  # Input
        self.T2 = 50+25  # Output
        self.F = 9  # The last dimension

        # Defining models
        self.model = JRTransformer(in_joint_size=self.T1 * self.F,
                                    feat_size=512,
                                    out_joint_size=self.T2 * 3,
                                    num_heads=args.num_heads, depth=args.depth).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=args.step_size, gamma=args.gamma)

        self.rc = args.rc

        # data
        dataset = Data(dataset='mocap_umpm', mode=0, device=args.device, transform=False)
        test_dataset1 = Data(dataset='mocap_umpm', mode=1, device=args.device, transform=False)
        print("Load Train set!")
        self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
        print("Load Test set!")
        self.test_dataloader = DataLoader(test_dataset1, batch_size=30, shuffle=False, drop_last=True)

        self.pretrain_path = args.pretrain_path
        # Training log
        self.log_dir = 'logs/'
        self.model_dir = 'd_' + str(args.depth) + '_' + \
                         'nh_' + str(args.num_heads) + '_' + \
                         datetime.now().strftime('%m%d%H%M') + '/'
        if not os.path.exists(self.log_dir + self.model_dir):
            os.makedirs(self.log_dir + self.model_dir)

    def test(self):
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

                J = 15  # joints
                input_data_original = input_data_original.reshape(B, N, T, J, -1)  # B,N,T,J,9
                output = output_seq.reshape(B, N, 26, J, -1).permute(0, 1, 3, 2, 4).reshape(B, -1, 26, 6)
                input_data_original = input_data_original.float().cuda()
                input_total = input_data_original.clone()
                input_data_original = input_data_original.permute(0, 2, 1, 3, 4)
                input_total = input_total.permute(0, 2, 1, 3, 4)

                batch_size = input_total.shape[0]

                if self.rc:
                    camera_vel = input_total[:, 1:40, :, :, 3:6].mean(dim=(1, 2, 3))
                    input_total[..., 3:6] -= camera_vel[:, None, None, None]
                    input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:6].cumsum(dim=1)

                # B, NxJ, T, 6
                input_data_original = input_data_original.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 50, 9)
                input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 50, 9)

                input_joint = input_total

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
                        pred_vel = pred_vel + camera_vel[:, None, None]
                    pred_motion_ = get_position(input_data_original[m], pred_vel)
                    pred_motion_ = pred_motion_.permute(0, 2, 1, 3)
                    pred_motion.append(pred_motion_)
                pred_motion = torch.cat(pred_motion,dim=1)

                """calculate metrics：JPE、APE、FDE"""
                B, NJ, T, D = pred_motion.shape
                gt_v = output[..., :3].reshape(B, -1, 15, T+1, 3).permute(0, 1, 3, 2, 4).cpu()  # B,N,T,J,3
                pred_v = pred_motion.reshape(B, -1, 15, T, D).permute(0, 1, 3, 2, 4).cpu()  # B,N,T,J,3

                n += 1
                jpe_err = JPE(pred_v, gt_v, frame_idx)
                ape_err = APE(pred_v, gt_v, frame_idx)
                fde_err = FDE(pred_v, gt_v, frame_idx)

                ape_err_total += ape_err
                jpe_err_total += jpe_err
                fde_err_total += fde_err

            with open(os.path.join(self.log_dir + self.model_dir, 'log.txt'), 'a+') as log:
                log.write(
                    'Test JPE:\t  200ms: {1:6.0f} | 400ms: {2:6.0f} | 600ms: {3:6.0f} | 800ms: {4:6.0f} | 1000ms: {5:6.0f}\n'.format(
                        "Our", jpe_err_total[0] / n, jpe_err_total[1] / n, jpe_err_total[2] / n, jpe_err_total[3] / n, jpe_err_total[4] / n))
                log.write(
                    'Test APE:\t  200ms: {1:6.0f} | 400ms: {2:6.0f} | 600ms: {3:6.0f} | 800ms: {4:6.0f} | 1000ms: {5:6.0f}\n'.format(
                        "Our", ape_err_total[0] / n, ape_err_total[1] / n, ape_err_total[2] / n, ape_err_total[3] / n, ape_err_total[4] / n))
                log.write(
                    'Test FDE:\t  200ms: {1:6.0f} | 400ms: {2:6.0f} | 600ms: {3:6.0f} | 800ms: {4:6.0f} | 1000ms: {5:6.0f}\n'.format(
                        "Our", fde_err_total[0] / n, fde_err_total[1] / n, fde_err_total[2] / n, fde_err_total[3] / n, fde_err_total[4] / n))
            loss = (jpe_err_total[0] / n + jpe_err_total[2] / n + jpe_err_total[4] / n) / 3.0
        return loss

    def train(self):
        start_time = time.time()
        steps = 0
        losses = []
        start_epoch = 0
        self.best_eval = 400

        for train_iter in range(start_epoch, self.num_epoch):
            print("Epoch:", train_iter)
            print("Time since start:", (time.time() - start_time) / 60.0, "minutes.")
            self.model.train()
            self.epoch = train_iter

            for i, data in enumerate(self.train_loader, 0):
                input_total, output_seq = data
                B, N, T, D = input_total.shape
                J = 15
                input_total = input_total.reshape(B, N, T, J, -1)  # B,N,T,J,D
                output = output_seq.reshape(B, N, 26, J, -1).permute(0, 1, 3, 2, 4).reshape(B, -1, 26, 6)
                input_total = input_total.float().cuda()
                input_total = input_total.permute(0, 2, 1, 3, 4) # B,T,N,J,D

                batch_size = input_total.shape[0]
                # delete camera movement
                if self.rc:
                    camera_vel = input_total[:, 1:40, :, :, 3:6].mean(dim=(1, 2, 3))  # B, 3
                    input_total[..., 3:6] -= camera_vel[:, None, None, None]
                    input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:6].cumsum(dim=1)

                input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 50, 9)  # （B,45,50,6） 45=NJ
                input_joint = input_total

                """ Interaction Perceptron """
                social = orientation_and_distance(input_joint)

                #  Processing input data: Split the input_joint of three people into individual x_input_joint.
                x_input_joint, y_input_joint = Input_divide(input_joint, social)

                # ground-truth
                gt_vel = torch.cat((input_total[..., 3:6], output[...,:-1, 3:6]), dim=-2)  # (B,NJ,T,D)
                gt_vel_x = gt_vel[:, :, :50]
                gt_vel_y = gt_vel[:, :, 50:]

                # Input individually
                recon_vel_ = []
                pred_vel_ = []
                for m in range(len(x_input_joint)):
                    pred_vel_x, pred_vel_aux_x = self.model(x_input_joint[m], y_input_joint[m])
                    pred_vel = pred_vel_x.permute(0, 2, 1, 3)
                    if self.rc:
                        pred_vel = pred_vel + camera_vel[:, None, None]
                    pred_vel = pred_vel.permute(0, 2, 1, 3)  # B，J，T，3
                    recon_vel, pred_vel, recon_vel_aux, pred_vel_aux = process_pred(pred_vel, pred_vel_aux_x)
                    recon_vel_.append(recon_vel)
                    pred_vel_.append(pred_vel)

                # loss
                loss_recon = distance_loss(torch.cat(recon_vel_,dim=1), gt_vel_x)
                loss_pred = distance_loss(torch.cat(pred_vel_,dim=1), gt_vel_y)
                loss = loss_pred * self.weight_loss_pred + loss_recon * self.weight_loss_recon

                self.opt.zero_grad()
                # Backward pass: compute gradient of the loss with respect to parameters.
                loss.backward()
                # Perform gradient clipping.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.opt.step()
                losses.append([loss.item()])
                steps += 1
            self.scheduler_model.step()

            print("Loss", np.array(losses).mean())

            with open(os.path.join(self.log_dir + self.model_dir, 'log.txt'), 'a+') as log:
                log.write('Epoch: {}, Train Loss: {},\n'.format(train_iter, np.array(losses).mean()))

            total_eval= self.test()

            if total_eval < self.best_eval:
                self.best_eval = total_eval
                self.best_model = self.model.state_dict()
                print('best_jpe_eval：{}'.format(self.best_eval))
                checkpoint = {
                    "net": self.model.state_dict(),
                }
                torch.save(checkpoint, self.log_dir + self.model_dir + 'best.pt')

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    trainer = Trainer(args)
    trainer.train()