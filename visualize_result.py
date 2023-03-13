import pickle
import numpy as np
import argparse
import os
import cv2
import matplotlib.pyplot as plt


def visualize_data(load_dir, save_dir, config):
    with open(os.path.join(load_dir, 'vis_dicts.pkl'), 'rb') as f:
        vis_dicts = pickle.load(f)

    print('total', len(vis_dicts))
    for idx, per_dict in enumerate(vis_dicts):
        # if idx > 10:
        #     break

        if per_dict['bev']['gt_traj'] is not None:
            visualize_per_view(per_dict['bev'], 'bev', save_dir, idx, config)

        if per_dict['fpv']['gt_traj'] is not None:
            visualize_per_view(per_dict['fpv'], 'fpv', save_dir, idx, config)

    print('ok')


def visualize_per_view(m_dict, tag, save_dir, idx, config):
    obs_tensor = m_dict['agents'][0].reshape(-1, 2)
    if config['with_gt']:
        gt_tensor = m_dict['gt_traj'].reshape(-1, 2)
    else:
        gt_tensor = None

    pred_list = []
    pred_trajs = m_dict['pred_traj']  # [6,30,2]
    for k in range(pred_trajs.shape[0]):
        pred_list.append(np.concatenate([pred_trajs[k, :, :], np.full((1, 2), np.nan)], axis=0))
    if len(pred_list) != 0:
        pred_tensor = np.concatenate(pred_list, axis=0).reshape(-1, 2)
    else:
        pred_tensor = np.zeros([0, 2]).reshape(-1, 2)

    pred_goals = m_dict['pred_traj'][:, -1, :]  # [6,2]

    neig_list = []
    for ped in m_dict['agents'][1:]:  # AV作为neig
        neig_list.append(np.concatenate([ped, np.full((1, 2), np.nan)], axis=0))
    if len(neig_list) != 0:
        neig_tensor = np.concatenate(neig_list, axis=0).reshape(-1, 2)
    else:
        neig_tensor = np.zeros([0, 2]).reshape(-1, 2)

    # lane process 没有差值
    lanes_list = []
    vis_lanes = m_dict['lanes']
    for lane in vis_lanes:
        lanes_list.append(np.concatenate([lane, np.full((1, 2), np.nan)], axis=0))
    if len(lanes_list) != 0:
        lanes_tensor = np.concatenate(lanes_list, axis=0).reshape(-1, 2)
    else:
        lanes_tensor = np.zeros([0, 2]).reshape(-1, 2)

    # center_list = []
    # polygons = mapping['polygons']
    # for lane in polygons:
    #     center_list.append(np.concatenate([lane, np.full((1, 2), np.nan)], axis=0))
    # if len(lanes_list) != 0:
    #     center_tensor = np.concatenate(center_list, axis=0).reshape(-1, 2)
    # else:
    #     center_tensor = np.zeros([0, 2]).reshape(-1, 2)

    # 可视化scores
    goals_2D = m_dict['goals_2D']  # np[d_goals,2]
    scores = m_dict['scores']  # np[d_goals]
    scores = np.exp(scores)
    colors = np.ones([scores.shape[0], 3])
    s_max = int(np.max(scores) * 10000) / 10000
    colors[:, 1] = 0.90 - 0.6 / s_max * scores  # 值越小，颜色越深

    # 作图
    fig, ax = plt.subplots()

    plt.plot(obs_tensor[:, 0], obs_tensor[:, 1],
             color='red', label='History', linestyle='-', linewidth=1, zorder=3)
    if config['with_gt']:
        plt.plot(gt_tensor[:, 0], gt_tensor[:, 1],
                 color='red', label='Ground Truth', linestyle='--', linewidth=1, zorder=3)
    if config['with_pred']:
        plt.plot(pred_tensor[:, 0], pred_tensor[:, 1],
                 color='green', label='Prediction', linestyle='--', linewidth=1, zorder=3)
    if config['with_goal']:
        plt.scatter(pred_goals[:, 0], pred_goals[:, 1],
                    color='orange', label='Predicted Goal', marker='*', s=16, zorder=4)
    plt.plot(neig_tensor[:, 0], neig_tensor[:, 1],
             color='blue', label='Neighbor', linestyle='-', linewidth=1, zorder=2)
    plt.plot(lanes_tensor[:, 0], lanes_tensor[:, 1],
             color='black', label='lane', linestyle='-', linewidth=0.5, zorder=1)
    # plt.plot(center_tensor[:, 0], center_tensor[:, 1],
    #          color='silver', label='center', linestyle='-', linewidth=0.5, zorder=1)
    if config['with_heatmap']:
        sort_id = np.argsort(-colors[:, 1])  # 按照颜色排序
        plt.scatter(goals_2D[sort_id, 0], goals_2D[sort_id, 1],
                    s=6, c=colors[sort_id, :], marker='o', zorder=1)

    plt.legend(ncol=3, loc='lower center')
    ax.set_aspect(1)
    plt.box(False)
    plt.axis('off')

    if tag == 'fpv':
        plt.xlim(0, 1920)
        plt.ylim(0, 1200)
        ax.invert_yaxis()
    else:
        plt.xlim(-40, 40)
        plt.ylim(-40, 40)

    save_file = os.path.join(save_dir, str(idx).zfill(6) + '_' + tag + '.png')

    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, dpi=400)

    plt.close('all')

    print(save_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--no_gt', action='store_true')
    parser.add_argument('--no_pred', action='store_true')
    parser.add_argument('--no_goal', action='store_true')
    parser.add_argument('--no_heat', action='store_true')
    parser.add_argument('--format', type=str, default='png')
    args = parser.parse_args()

    checkpoint_path = args.load_model_path
    args_path = os.path.split(checkpoint_path)[0]
    load_dir = os.path.join(args_path, 'eval_results')
    save_dir = os.path.join(load_dir, 'fig')
    config = {'with_gt': not args.no_gt, 'with_pred': not args.no_pred,
              'with_goal': not args.no_goal, 'with_heatmap': not args.no_heat}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('--------start visualizing data--------------')
    visualize_data(load_dir, save_dir, config)


if __name__ == '__main__':
    main()