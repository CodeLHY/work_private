'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''
import os
import os.path
import logging
import sys
import math
import lmdb
import tqdm
import numpy as np
import random
import msgpack_numpy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
from provider import *


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()
BASE_DIR = "../pointnet2_tf"

def extract_names_from_txt(path_txt):
    """
    :param path_txt:  modelnet40_name 文件所在的地方
    :return: 返回modelnet这个数据集用到的所有的数据的种类的名称
    """
    names = []
    with open(path_txt, 'r') as file:
        for line in file:
            name = line.rstrip()
            names.append(name)
    return names
def get_train_test_paths(path_txt, dict_cls):
    """
    得到所有 训练/测试 集数据的paths
    :param path_txt: 用于记录需要被取出的数据的名称的txt文件
    :return: [(cls, paths)]
    """
    paths = []
    files = extract_names_from_txt(path_txt)
    for file in files:
        category = "_".join(file.split('_')[0:-1])
        cls = dict_cls[category]
        path = os.path.join(os.path.dirname(path_txt), category, file+".txt")
        paths.append((cls, path))
    return paths
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ModelNetDataset_lmdb():
    def __init__(self, root, batch_size=32, npoints=1024, split='train', normalize=True, shuffle=None):
        super().__init__()
        self.cls_paths_data = None
        self._lmdb_env = None
        self.num_points = npoints
        self.num_channel = 6
        self.global_step = 0
        self.batch_size = batch_size
        # 获取所有的标签, 将其映射为一个数字标签
        path_names = os.path.join(root, "modelnet40_shape_names.txt")
        names = extract_names_from_txt(path_names)
        dict_name_label = dict(zip(names, range(len(names))))
        # 获取用于训练或者测试的数据的path   self.cls_paths_data
        if split=='train':
            # 记录用于训练的数据的txt文件的地址
            path_train_txt = os.path.join(root, "modelnet40_train.txt")
            # 获取所有用于训练的数据[(cls, paths)]
            paths_train = get_train_test_paths(path_train_txt, dict_name_label)
            self.cls_paths_data = paths_train
        else:
            # 记录用于测试的数据的txt文件的地址  [(cls, path_txt)]
            path_test_txt = os.path.join(root, "modelnet40_test.txt")
            # 获取所有用于训练的数据[(cls, paths)]
            paths_test = get_train_test_paths(path_test_txt, dict_name_label)
            self.cls_paths_data = paths_test
            # 使用lmdb包来提高数据的读取速度
        print("Converted to LMDB for faster dataloading while training")
        self.dir_cache = os.path.join(os.path.dirname(__file__), "cache_modelnet40")
        if not os.path.exists(self.dir_cache):
            os.mkdir(self.dir_cache)
        if not os.path.exists(os.path.join(self.dir_cache, split)):
            with lmdb.open(os.path.join(self.dir_cache, split), map_size=1 << 36) as lmdb_env, lmdb_env.begin(
                    write=True) as txn:
                for i in tqdm.trange(len(self.cls_paths_data)):
                    fn = self.cls_paths_data[i]
                    point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
                    if normalize:
                        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
                    cls = self.cls_paths_data[i][0]
                    cls = int(cls)
                    txn.put(
                        str(i).encode(),
                        msgpack_numpy.packb(
                            dict(pc=point_set, lbl=cls), use_bin_type=True
                        ),
                    )
        self._lmdb_file = os.path.join(self.dir_cache, split)
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]
    # 是否随机选用数据也就是是否将idx打乱
        if shuffle is None:
            if split == 'train':
                self.shuffle = True
            else:
                self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
        rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        jittered_data = provider.random_scale_point_cloud(rotated_data[:, :, 0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        return provider.shuffle_points(rotated_data)

    def _get_item(self, idx):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )
        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)
        point_set = ele["pc"]
        if self.num_points is not None:
            # order = [i for i in range(point_set.shape[0])]
            # random.shuffle(order)
            # point_set = point_set[order, :]  # todo: 这里没有先打乱再取前numpoints个而是像pointnet里面一样，直接取出前num_points个
            point_set = point_set[0:self.num_points, :]

        return point_set, ele['lbl']

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return self._len

    def reset(self):
        self.idxs = np.arange(0, self._len)
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (self._len + self.batch_size - 1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, self._len)
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.num_points, self.num_channel))
        batch_label = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps, cls = self._get_item(self.idxs[i + start_idx])
            batch_data[i] = ps
            batch_label[i] = cls
        self.batch_idx += 1
        if augment: batch_data = self._augment_batch_data(batch_data)
        return batch_data, batch_label

# if __name__ == '__main__':
# d = ModelNetDataset(root = '../data/modelnet40_normal_resampled', split='test')
# print(d.shuffle)
# print(len(d))
# import time
# tic = time.time()
# for i in range(10):
#     ps, cls = d[i]
# print(time.time() - tic)
# print(ps.shape, type(ps), cls)
#
# print(d.has_next_batch())
# ps_batch, cls_batch = d.next_batch(True)
# print(ps_batch.shape)
# print(cls_batch.shape)
