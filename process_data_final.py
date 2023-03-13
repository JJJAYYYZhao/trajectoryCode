import os
import argparse
import numpy as np
import pickle
import zlib
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_tag', default='argo', type=str)  # argo or ns
    parser.add_argument('--split', action='store_true')
    args = parser.parse_args()

    dataset_tag = args.dataset_tag

    if dataset_tag == 'argo':
        data_dir = '../argoverse-data/processed'
    elif dataset_tag == 'ns':
        data_dir = '../nuscenes-data/processed'
    else:
        data_dir = 'none'

    if args.split and (dataset_tag != 'ns'):  # ns不需要split
        data_split(data_dir)



def data_split(data_dir):
    for type1, type2 in [('train', 'full'), ('train', 'subset'),
                         ('val', 'full'), ('val', 'subset'),
                         ('test', 'full')]:
        print('split data in {} {}'.format(type1, type2))

        save_dir = os.path.join(data_dir, 'splited', type1 + '_' + type2)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load data
        file = os.path.join(data_dir, type1 + '_' + 'bev' + '_' + 'traj.pkl')
        print(file)
        with open(file, 'rb') as f:
            data_list1 = pickle.load(f)
        # file = os.path.join(data_dir, type1 + '_' + 'fpv' + '_' + 'traj.pkl')
        # print(file)
        # with open(file, 'rb') as f:
        #     data_list2 = pickle.load(f)

        # load index
        if type1 == 'test':
            index_list = [True] * len(data_list1)
        else:
            if type2 == 'full':
                index_list = [True] * len(data_list1)
            else:
                index_list = (np.linspace(0, len(data_list1)-1, len(data_list1)) % 10 == 0).tolist()

        assert len(data_list1) == len(index_list)
        # assert len(data_list2) == len(index_list)
        data_list1 = [y for x, y in zip(index_list, data_list1) if x]  # 这一边仅仅保存符合条件的，然后文件id是连续的
        # data_list2 = [y for x, y in zip(index_list, data_list2) if x]

        for id in tqdm(range(len(data_list1))):
            instance1 = pickle.loads(zlib.decompress(data_list1[id]))
            # instance2 = pickle.loads(zlib.decompress(data_list2[id]))
            # mapping = {'bev': instance1, 'fpv': instance2}
            mapping = instance1
            mapping = zlib.compress(pickle.dumps(mapping))

            # save
            save_file = os.path.join(save_dir, str(id).zfill(8) + '.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(mapping, f)



if __name__ == '__main__':
    main()