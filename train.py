import torch
import os
import argparse
import time
import json
import numpy as np
import sys
sys.path.append(os.path.join(os.getcwd(), 'external_cython'))
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# from utils import log_help
from configs import cfg
from tensorboardX import SummaryWriter
from utils.func_lab import make_dir, get_metric_dict, print_info, get_miss_rate, MyLogger, get_from_mapping
from dataloader.argo_loader import make_dataloader
from modeling.my_class import TFTraj
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'  # 多卡训练记得通过这里控制device


def main():  # todo: 把所有np的reshape改成[:, None, :]这样的
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/szj.yml', type=str)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    save_dir = os.path.join('..', cfg.save_dir, cfg.save_tag, 'train_' + time.strftime("%Y_%m_%d_%H_%M"))
    make_dir(save_dir)

    # log_help.set_logger(os.path.join(save_dir, 'train_' + time.strftime("%d_%m_%Y_%H_%M_%S") + '.log'))
    logger = MyLogger(os.path.join(save_dir, 'train_' + time.strftime("%d_%m_%Y_%H_%M_%S") + '.log'))

    if torch.cuda.is_available():
        device = cfg.device
    else:
        device = 'cpu'

    if torch.cuda.device_count() < cfg.distributed:
        print('warring, not enough gpu device')
        cfg.distributed = 1

    # change cfg
    if cfg.modality != 'both':
        cfg.MODEL.cross_type = 0
        cfg.MODEL.share_weight = True
        cfg.MODEL.cross_enc = False
    if cfg.MODEL.cross_type == 0:
        assert cfg.modality != 'both'
    if not cfg.MODEL.share_weight:
        assert cfg.modality == 'both'

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f)

    print_info(args, cfg, device, save_dir, logger)

    if cfg.distributed == 1:
        main_process_nodist(cfg, args, device, save_dir, logger)
    else:
        logger.info('warning: distributed > 1')


def main_process_nodist(cfg, args, device, save_dir, logger):
    model = TFTraj(cfg, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.lr)

    if cfg.SOLVER.scheduler_lr_type == 'm':
        scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.milestones, gamma=0.1)
    elif cfg.SOLVER.scheduler_lr_type == 's':
        scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.step_size, gamma=0.1)
    else:
        print('error scheduler_lr_type')
        scheduler_lr = None

    metric_dict_best = {'ade': [0, 1000], 'fde': [0, 1000]}

    # load modeling
    if args.load_model:
        checkpoint_path = args.load_model_path
        if not os.path.exists(checkpoint_path):
            train_from_epoch = 0
            logger.info('train from {} ---- fail loading from {}'.format(train_from_epoch, checkpoint_path))
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
            model.load_state_dict(checkpoint['model'])

            train_from_epoch = int(checkpoint_path[-8:-4])
            logger.info('train from {} ---- loading from {}'.format(train_from_epoch, checkpoint_path))
    else:
        train_from_epoch = 0
        logger.info('train from {}'.format(train_from_epoch))

    # load data
    logger.info('preparing train dataloader')
    train_dataloader = make_dataloader(cfg, 'train')

    logger.info('preparing test dataloader')
    val_dataloader = make_dataloader(cfg, 'val')

    if cfg.test_include:
        logger.info('preparing test dataloader')
        test_dataloader = make_dataloader(cfg, 'val')
    else:
        test_dataloader = None

    # debug
    # import pickle
    # with open('mapping_ttt.pkl', 'rb') as f:
    #     data_test = pickle.load(f)
    # train_dataloader = [[{'bev': data_test}, ], ]
    # val_dataloader = [[{'bev': data_test}, ], ]

    writer = SummaryWriter(os.path.join(save_dir, 'summary_' + time.strftime("%d_%m_%Y_%H_%M_%S")))

    # main loop
    for epoch in range(train_from_epoch + 1, cfg.num_epochs + 1):
        #----train----
        train(cfg, device, epoch, train_dataloader, optimizer, scheduler_lr, writer, logger, model)

        #----val----
        metric = test(cfg, device, epoch, val_dataloader, writer, logger, model, 'val')
        for metric_type in list(metric.keys()):
            if metric_dict_best[metric_type][1] > metric[metric_type]:
                metric_dict_best[metric_type][0] = epoch
                metric_dict_best[metric_type][1] = metric[metric_type]

        #----test----
        if cfg.test_include:
            pass

        # print best test and val
        for metric_type in list(metric_dict_best.keys()):
            logger.info('BEST {} = {}  at epoch {}'.format(metric_type, metric_dict_best[metric_type][1], metric_dict_best[metric_type][0]))
        logger.info('----------------------------')

        # save
        checkpoint = {'epoch': epoch,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_lr_state_dict': scheduler_lr.state_dict(),
                      'model': model.state_dict()}
        checkpoint_path = os.path.join(save_dir, 'model_epoch_' + str(epoch).zfill(4) + '.tar')
        if cfg.save_model and (epoch % cfg.save_interval == 0):
            torch.save(checkpoint, checkpoint_path)
            logger.info('save modeling to file: {}'.format(save_dir))


def train(cfg, device, epoch, dataloader, optimizer, scheduler_lr, writer, logger, model):
    epoch_loss = 0
    model.train()
    infer = False
    metric_dict = {'ade': [], 'fde': []}

    for step, batch in enumerate(dataloader):
        time_start = time.time()
        loss, y_pred, output_dict = model(batch, infer, cfg.MODEL.num_samples)  # [len,K*bs,2]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        time_end = time.time()

        writer.add_scalar('total_loss/iter', loss.item(), step + (epoch - 1) * len(dataloader))

        logger.info(
            '{}/{}, [epoch {}/{}], loss = {:.6f}, lr = {}, time/batch = {:.3f}'.format(
                step, len(dataloader), epoch, cfg.num_epochs,
                loss.item(), optimizer.param_groups[0]['lr'], time_end - time_start))

        epoch_loss += loss.item()

        pos_pred = y_pred.detach().to('cpu').numpy()  # [len,K*bs,2]
        pos_gt = output_dict['y_gt']  # [len,bs,2]

        metric_ade_fde = get_metric_dict(pos_pred, pos_gt, cfg.MODEL.num_samples)  # ml的num_sample取1
        metric_dict['ade'].append(metric_ade_fde['ade'])
        metric_dict['fde'].append(metric_ade_fde['fde'])

    scheduler_lr.step()

    epoch_loss = epoch_loss / len(dataloader)
    writer.add_scalar('loss/epoch', epoch_loss, epoch)
    logger.info('finish train epoch {}, epoch loss = {}'.format(epoch, epoch_loss))

    logger.info('----------------------------')
    for key in metric_dict.keys():
        # miss_rates = (get_miss_rate(li_FDE[key], dis=2.0), get_miss_rate(li_FDE[key], dis=4.0), get_miss_rate(li_FDE[key], dis=6.0))
        logger.info('metric = {} ADE = {}, FDE = {}'.format(
            key, np.mean(metric_dict[key]) if len(metric_dict[key]) > 0 else None,
            np.mean(metric_dict[key]) if len(metric_dict[key]) > 0 else None))
    logger.info('----------------------------')

    return epoch_loss


def test(cfg, device, epoch, dataloader, writer, logger, model, val_test_flag):
    model.eval()
    infer = True
    metric_dict = {'ade': [], 'fde': []}

    for step, batch in enumerate(dataloader):
        _, y_pred, output_dict = model(batch, infer, cfg.MODEL.num_samples)  # [len,K*bs,2]

        pos_pred = y_pred.detach().to('cpu').numpy()  # [len,K*bs,2]
        pos_gt = output_dict['y_gt']  # [len,bs,2]

        metric_ade_fde = get_metric_dict(pos_pred, pos_gt, cfg.MODEL.num_samples)  # ml的num_sample取1
        metric_dict['ade'].append(metric_ade_fde['ade'])
        metric_dict['fde'].append(metric_ade_fde['fde'])

    logger.info('----------------------------')
    for key in metric_dict.keys():
        # miss_rates = (get_miss_rate(li_FDE[key], dis=2.0), get_miss_rate(li_FDE[key], dis=4.0), get_miss_rate(li_FDE[key], dis=6.0))
        logger.info('metric = {} ADE = {}, FDE = {}'.format(
            key, np.mean(metric_dict[key]) if len(metric_dict[key]) > 0 else None,
            np.mean(metric_dict[key]) if len(metric_dict[key]) > 0 else None))
    logger.info('----------------------------')

    # 求batch之间的平均值，之后log
    metric_dict_epoch = dict()
    for metric_type in metric_dict.keys():
        metric_dict_epoch[metric_type] = np.mean(np.array(metric_dict[metric_type]))

    for metric_type in list(metric_dict_epoch.keys()):
        writer.add_scalar(metric_type, metric_dict_epoch[metric_type], epoch)

    logger.info('finish {} epoch {}'.format(val_test_flag, epoch))
    for metric_type in list(metric_dict_epoch.keys()):
        logger.info(metric_type + ' = {}'.format(metric_dict_epoch[metric_type]))
    logger.info('----------------------------')

    return metric_dict_epoch




if __name__ == '__main__':
    main()