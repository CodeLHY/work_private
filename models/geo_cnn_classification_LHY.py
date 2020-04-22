import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tensorflow as tf
import numpy as np
import tf_utils
from knn_mlp_utils import mlp_knn
from geo_utils_LHY import geoconv
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))  #todo:https://github.com/charlesq34/pointnet2/issues/41 tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    # todo: 所有的运算中weight_decay都没有开启，如果觉得泛化能力不够的话，打开一下试一试
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])  # todo : here is the change according to https://github.com/charlesq34/pointnet2/issues/41
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])  # todo : too
    # ==========================layer1_branch_1=============使用和pointnet一样方法，找出周围的16个点
    l1_points_1 = mlp_knn(k=16,xyz=l0_xyz, points=l0_points, is_training=is_training, bn_decay=bn_decay, scope='l1_branch_1', pooling='max')  # b*n*384
    # logger.debug("l1_points_1:", l1_points_1.shape) b*n*384
    # ==========================layer1_branch_2====================================
    # Set abstraction layers
    l1_points_2 = geoconv(feat=l0_points, xyz=l0_xyz, num_outputs=128, bypass_num_outputs=64, radius=0.05, decay_radius=0.15, bn=True, is_training=is_training, scope='geoconv_1', bn_decay=bn_decay, activation_fn=tf.nn.relu, delta=False)
    # logger.debug("l1_points_2:", l1_points_2.shape)
    l1_points_2 = tf.expand_dims(l1_points_2, axis=[-1])  # b*n*128*1
    # logger.debug("l1_points_2:", l1_points_2.shape)
    l1_points_2 = tf.transpose(l1_points_2, [0, 2, 1, 3])  # b*128*n*1
    # logger.debug("l1_points_2:", l1_points_2.shape)
    l1_points_2 = tf_utils.conv2d(l1_points_2, 256, [1, 1],padding='VALID', stride=[1, 1], data_format='NCHW',bn=True, is_training=is_training,scope='FC_l1_branch2_1', bn_decay=bn_decay,activation_fn=tf.nn.relu)  # b*256*n*1
    # logger.debug("l1_points_2:", l1_points_2.shape)
    l1_points_2 = tf.transpose(l1_points_2, [0, 2, 1, 3])  # b*n*256*1
    # logger.debug("l1_points_2:", l1_points_2.shape)
    l1_points_2 = tf.squeeze(l1_points_2,axis=[-1])  # b*n*256
    # logger.debug("l1_points_2:", l1_points_2.shape)
    l1_points_2 = geoconv(feat=l1_points_2, xyz=l0_xyz, num_outputs=512, bypass_num_outputs=64, radius=0.05,
                        decay_radius=0.3, bn=True, is_training=is_training, scope='geoconv_2', bn_decay=bn_decay,
                        activation_fn=tf.nn.relu, delta=False)  # b*n*512
    # logger.debug("l1_points_2:", l1_points_2.shape)
    l1_points = tf.concat([l1_points_1,l1_points_2],axis=-1)  # b*n*896
    # logger.debug("l1_points:", l1_points.shape)
    # ===============================layer2=========================================
    l2_points = geoconv(feat=l1_points, xyz=l0_xyz, num_outputs=768, bypass_num_outputs=64, radius=0.05, decay_radius=0.6, bn=True, is_training=is_training, scope='geoconv_3', bn_decay=bn_decay, activation_fn=tf.nn.relu, delta=False)
    # logger.debug("l2_points:", l2_points.shape)
    l2_points = tf.expand_dims(l2_points,axis=[-1])  # b*n*768*1
    # logger.debug("l2_points:", l2_points.shape)
    l2_points = tf.transpose(l2_points,[0,2,1,3])  # b*768*n*1
    # logger.debug("l2_points:", l2_points.shape)
    l2_points = tf_utils.conv2d(l2_points, 2048, [1,1],padding='VALID', stride=[1, 1], data_format='NCHW',bn=True, is_training=is_training,scope='FC_l2_branch2_1', bn_decay=bn_decay,activation_fn=tf.nn.relu)
    # b*2048*n*1
    # logger.debug("l2_points:", l2_points.shape)
    l2_points = tf.squeeze(l2_points, axis=[-1])  # b*2048*n
    # logger.debug("l2_points:", l2_points.shape)
    l2_points = tf.reduce_max(l2_points,axis=[-1],keep_dims=False,name='l2_maxpooling')  # b*2048
    # ===============================classification================================
    output = tf_utils.fully_connected(inputs=l2_points,num_outputs=40,scope='classification',activation_fn=None,bn=False,bn_decay=None,is_training=is_training)
    # todo:分类层没有用bn
    # b*40
    # logger.debug("output:", output.shape)
    return output


def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)  # 计算整体的均值（一个batch）
    tf.summary.scalar('classify loss', classify_loss)  # 添加一个监控的元素
    tf.add_to_collection('losses', classify_loss)  # 将这个loss加入到losses这个collection中
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)