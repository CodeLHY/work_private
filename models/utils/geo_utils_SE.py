""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/grouping'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/3d_interpolation'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/geoconv'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point

from tf_geoconv import aggregate  # 在这里，已经被加入到路径里面了sys.path.append(os.path.join(BASE_DIR, 'tf_ops/geoconv'))
from tf_utils import perceptron, batch_norm_for_fc, batch_norm_for_conv1d
import tensorflow as tf
import numpy as np


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1, 1, nsample)), (batch_size, 1, 1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def geoconv(feat, xyz, num_outputs, bypass_num_outputs,
            radius, decay_radius, bn=True, is_training=False,
            scope=None, bn_decay=None, activation_fn=tf.nn.relu,
            delta=False,dropout_self=0, dropout_edge_1=0, dropout_dege_2=0):
    ''' GeoCNN Geo-Conv
        Input:
            feat: (batch_size, num_point, input_channels) TF tensor
            points: (batch_size, num_point, 3) TF tensor
            num_outputs: the count of output channels
            bypass_num_outputs: the count of output channels of bypass
            radius: the inner radius of local ball of each point.
            decay radius: the outer radius of local ball of each point
            ...
    '''
    with tf.variable_scope(scope) as sc:
        feat_self = tf.layers.dropout(feat, rate=dropout_self, training=is_training)  # dropout_self
        self = perceptron(feat_self, num_outputs, scope='self', is_training=is_training, bn=False, activation_fn=None)
        feat_edge = tf.layers.dropout(feat, rate=dropout_edge_1, training=is_training)  # dropout_edge_1
        mutual = perceptron(feat_edge, bypass_num_outputs * 6, scope='mutual', is_training=is_training, bn=False,
                            activation_fn=None)
        #  通过mlp计算出其在六个方向上的特征==》(计算geo特征，从B*N*C==》B*N*6*C，也即每一个点的每一个channel的特征都是分解为六个方向的特征，在aggregate里面会对六个方向的其中三个通过cos平方以及边缘点的计算出的权重进行聚合从何回归到B*N*C)
        ag, _ = aggregate(mutual, xyz, radius, decay_radius, delta)
        b, n, bc = ag.get_shape().as_list()
        _, _, c = self.get_shape().as_list()
        ag = tf.reshape(ag, (-1, bc))
        self = tf.reshape(self, (-1, c))

        if bn:
            ag = batch_norm_for_fc(ag, is_training, bn_decay, scope='mutual_bn')
        if activation_fn is not None:
            ag = activation_fn(ag)

        ag = tf.reshape(ag, (b, n, bc))
        ag = tf.layers.dropout(ag, rate=dropout_dege_2, training=is_training)  # dropout_edge_2
        ag = perceptron(ag, num_outputs, scope='enlarge', is_training=is_training, bn=False, activation_fn=None)
        ag = tf.reshape(ag, (-1, c))

        outputs = self + ag

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, scope='enlarge_bn')
        if activation_fn is not None:
            outputs = activation_fn(outputs)

        outputs = tf.reshape(outputs, (b, n, c))

        return outputs

