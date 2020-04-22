'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from data.modelnet40_dataset_lmdb import ModelNetDataset_lmdb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import tf_utils
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger()



parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=0, help='a unique mark of this training')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--train_with_aug', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
HOSTNAME = socket.gethostname()
NUM_CLASSES = 40
MODEL = importlib.import_module("geo_cnn_classification")  # import network module
Data_dir = "../pointnet2_tf/data"
# ===============================设置训练的log====================================
Base_dir = os.path.dirname(__file__)
Logs_dir = os.path.join(Base_dir, "Logs")  # 用于存放所有训练结果的总文件夹
if not os.path.exists(Logs_dir):
    os.mkdir(Logs_dir)
Log_dir = os.path.join(Logs_dir, FLAGS.name)  # 用于存放本次训练的所有内容的文件夹
if not os.path.exists(Log_dir):
    os.mkdir(Log_dir)
Result_dir = os.path.join(Log_dir, "results")  # 用于存放本次训练的所有结果
if not os.path.exists(Result_dir):
    os.mkdir(Result_dir)
log_realtime =open(os.path.join(Log_dir, "log.txt"), 'w')  # 打开一个记录实时情况的文件
def write_log(out_str):
    log_realtime.write(out_str+'\n')
    log_realtime.flush()
    print(out_str)
logger.debug("whether using nornal:"+ str(FLAGS.normal))


assert (NUM_POINT <= 10000)
DATA_PATH = os.path.join(Data_dir, 'modelnet40_normal_resampled')
TRAIN_DATASET = ModelNetDataset_lmdb(root=DATA_PATH, batch_size=BATCH_SIZE, npoints=NUM_POINT, split='train', normalize=True, shuffle=None)
TEST_DATASET = ModelNetDataset_lmdb(root=DATA_PATH, batch_size=BATCH_SIZE, npoints=NUM_POINT, split='test', normalize=True, shuffle=None)




def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)  # 要求输入是BCWH的形式，由于这里的输入
            # 是一个球中的点，因此可以将点数类比通道数，一个点的特征长度代表WH
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)  # 有名叫batch的变量的话就返回给batch，没有的话就新建一个
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)  # 在tensorboard上添加一个标量监控值

            # Get model and loss
            pred = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')  # 将这个collection里面的所有数据求和
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)  # tf.cast数据类型转化
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)

            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)  # global_step传入的参数会在每次优化后+1

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)  # 自动与默认的图相关联

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(Log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(Log_dir, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()  # 定义初始化操作
        sess.run(init)  # 执行图中数据的初始化

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            write_log('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(Log_dir, "model.ckpt"))
                write_log("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    # logger.debug()
    write_log(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=FLAGS.train_with_aug)
        # batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)  # run(fetchs, feed_dict, option, run_metadata).
        train_writer.add_summary(summary, step)  # 保存所有之前放在merged这个域中的监控的参数，还有step
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val

        if (batch_idx + 1) % 50 == 0:
            write_log(' ---- batch: %03d ----' % (batch_idx + 1))
            write_log('mean loss: %f' % (loss_sum / 50))
            write_log('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TEST_DATASET.num_channel))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    write_log(str(datetime.now()))
    write_log('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
    total_acc = total_correct / float(total_seen)
    v = tf.Summary.Value(tag="acc_val_epoch", simple_value=total_acc)
    s = tf.Summary(value=[v])
    test_writer.add_summary(s, step)
    # tf.summary.scalar("acc_val_epoch", total_acc)  # add a monitored scalar 源码只给了每个step的acc，这个波动太大了
    write_log('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    write_log('eval accuracy: %f' % (total_correct / float(total_seen)))
    write_log('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return total_correct / float(total_seen)


if __name__ == "__main__":
    write_log('pid: %s' % (str(os.getpid())))
    train()
    log_realtime.close()
