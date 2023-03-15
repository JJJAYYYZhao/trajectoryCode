import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from modeling.vectornet import VectorNet
from modeling.Transformer_Lib import TransDecoder
from utils.func_lab import get_from_mapping, batch_init_origin, to_origin_coordinate, get_dis_point_2_points
import json
# from utils.calibration import Calibration
# from utils.calibration_ns import Calibration_NS


class TFTraj(nn.Module):
    def __init__(self, cfg, device):
        super(TFTraj, self).__init__()
        self.cfg = cfg
        self.device = device
        self.modality = cfg.modality
        self.pred_len = cfg.DATA.pred_len
        self.tgt_input_dim = 2
        self.output_dim = 2

        self.tf_encoder = VectorNet(cfg, device)
        self.tf_decoder = TransDecoder(self.tgt_input_dim, self.output_dim, d_model=cfg.MODEL.hidden_size)

    def forward(self, batch, infer, num_samples, kl_weight=10.0):
        mapping = batch
        labels_np = np.array(get_from_mapping(mapping, 'labels')).transpose([1, 0, 2])  # list=[bs]=[30,2] -> [len,bs,2]
        bs = labels_np.shape[1]
        if not infer:
            labels = torch.tensor(labels_np, dtype=torch.float, device=self.device)  # [len,bs,2]
        else:
            labels = None
        #his_traj_ = get_from_mapping(mapping, 'agents')  # list[bs]=list[n]=[?,2]
        #his_traj_np = [lll[0] for lll in his_traj_]
        #his_traj_np = np.array(his_traj_np)  # [bs,20,2]
        # last_state = torch.tensor(his_traj_np[:, -1, :], dtype=torch.float, device=self.device)  # [bs,2]
        _, _, _, _, h_states = self.tf_encoder(mapping, self.device, infer)  # [bs,max_p_n,128]
        # tgt = h_states[:, 0:1, :]  # [bs,1,dim]
        if self.cfg.MODEL.tgt_random:
            # inital the target query in random
            tgt = torch.randn(bs, num_samples * self.pred_len, 2).to(self.device)  # [bs,k*len,2]
        else:
            # inital the target query by 000
            tgt = torch.ones(bs, num_samples * self.pred_len, 2).to(self.device)  # [bs,k*len,2]

        y_pred = self.tf_decoder(h_states, None, tgt, None, True)  # [bs,1*len,2]
        y_pred = y_pred.reshape(bs, num_samples, self.pred_len, 2).permute(2, 1, 0, 3).reshape(self.pred_len, -1, 2)  # [len,K*bs,2]

        if not infer:
            y_gt = labels
            recons_loss = recons_loss_func(self.cfg.DATA.pred_len, y_pred, y_gt)
            loss = recons_loss
        else:
            loss = None

        output_dict = {'y_gt': labels_np}  # 记录label、lane等信息

        return loss, y_pred, output_dict  # [len,K*bs,2]


def get_l2_loss(pred, gt):
    # pred=[bs,bs',2] gt=[bs,bs',2]
    loss = torch.sqrt(torch.sum((pred - gt) ** 2, dim=2))
    return loss  # [bs,bs']


def recons_loss_func(pred_len, y_pred, y_gt):
    # y_pred=[pred_len,K*bs,2], y_gt=[pred_len,bs,2]
    K = int(y_pred.shape[1] / y_gt.shape[1])

    y_gt = y_gt.unsqueeze(1).repeat(1, K, 1, 1).reshape(pred_len, -1, 2)

    recons_loss_all = get_l2_loss(y_pred, y_gt).reshape(pred_len, K, -1)  # [pred_len,K*bs]→[pred_len,K,bs]

    recons_loss = torch.mean(torch.min(torch.mean(recons_loss_all, dim=0), dim=0).values)

    return recons_loss
