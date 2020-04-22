import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
mama_dir = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(mama_dir, 'tf_ops/grouping'))
from tf_grouping import query_ball_point, group_point, knn_point
import tensorflow as tf
import numpy as np
import tf_utils
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()
def mlp_knn(k,xyz, points, is_training, bn_decay, scope, pooling='max'):
    '''

    :param k: 取最邻近的几个点
    :param xyz: 点集中每个点的坐标
    :param points: 点集中每个点的特征
    :param is_training: 是否处于训练的过程中
    :param bn_decay:
    :param scope:
    :param pooling: 使用什么方式的pooling
    :return:
    '''
    with tf.variable_scope(scope) as sc:
        dists, top_k = knn_point(k, xyz, xyz)  # 从数据集中找出每一个点最邻近的k个点 b*n*k
        top_k_features = group_point(points, top_k)  # 找出topk对应的feature b*n*k*3
        # logger.debug(top_k_features.shape)
        top_k_xyz = group_point(points, top_k)  #找出topk对应的xyz坐标 b*n*k*3
        # logger.debug(top_k_xyz.shape)
        top_k_all = tf.concat([top_k_xyz,top_k_features], axis=-1)  # b*n*k*6
        # logger.debug(top_k_all.shape)
        top_k_all = tf.transpose(top_k_all,[0,3,1,2])  # b*6*n*k
        # logger.debug(top_k_all.shape)
        out = tf_utils.conv2d(top_k_all, 64,[1,1],padding='VALID', stride=[1, 1], data_format='NCHW', bn=True,
                                        is_training=is_training, scope='FC_l1_branch1_1', bn_decay=bn_decay,
                                        activation_fn=tf.nn.relu)# b*64*n*k
        # logger.debug(out.shape)
        out = tf_utils.conv2d(out, 128,[1,1],padding='VALID', stride=[1, 1], data_format='NCHW', bn=True,
                                        is_training=is_training, scope='FC_l1_branch1_2', bn_decay=bn_decay,
                                        activation_fn=tf.nn.relu)# b*128*n*k
        # logger.debug(out.shape)
        out = tf_utils.conv2d(out, 384, [1, 1], padding='VALID', stride=[1, 1], data_format='NCHW', bn=True,
                              is_training=is_training, scope='FC_l1_branch1_3', bn_decay=bn_decay,
                              activation_fn=tf.nn.relu)  # b*384*n*k
        # logger.debug(out.shape)
        out = tf.transpose(out, [0,2,3,1])  # b*n*k*384
        # logger.debug(out.shape)
        if pooling=='max':
            out = tf.reduce_max(out, axis=[2], keep_dims=False,name="maxpooling")  # b*n*384
        elif pooling=='mean':
            out = tf.reduce_mean(out, axis=[2], keep_dims=False, name="maxpooling")  # b*n*384
        # logger.debug(out.shape)
        return out
