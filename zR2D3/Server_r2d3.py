# coding: utf-8

import sys

sys.path.append("/opt/tiger/test_ppo")

import tensorflow as tf
import numpy as np
import os
import logging
from collections import OrderedDict
import glob
import argparse
from multiprocessing import Queue, Process
import zmq

from utils import unpack

from zR2D3_boots.Worker_r2d3 import Worker_Q
from zR2D3_boots.Trainer_r2d3 import Model

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def run(**kwargs):
    tmplimit = 512

    server_id = kwargs.get("server_id", 0)
    num_servers = kwargs.get("num_servers", 1)

    address = "ipc:///tmp/databack%d" % server_id

    SCRIPT_DIR = kwargs.get("SCRIPT_DIR")
    BASE_DIR = kwargs.get("BASE_DIR")
    CKPT_DIR = kwargs.get("CKPT_DIR")
    DATA_DIR = kwargs.get("DATA_DIR")

    logging.basicConfig(
        filename=os.path.join(
            BASE_DIR, "Serverlog"),
        level="INFO")

    frames = kwargs.get("frames", 1)
    workers = kwargs.get("workers", 16)
    parallel = kwargs.get("worker_parallel", 4)
    n_step = kwargs.get("n_step", 5)
    gamma = kwargs.get("gamma", 0.99)
    seqlen = kwargs.get("seqlen", 80)
    burn_in = kwargs.get("burn_in", 40)
    epsilon = kwargs.get("epsilon", 0.4)
    epsilon_power = kwargs.get("epsilon_power", 8)

    games = ["SuperMarioBros-1-1-v0",
             "SuperMarioBros-2-1-v0",
             "SuperMarioBros-4-1-v0",
             "SuperMarioBros-5-1-v0"]

    games = ["SuperMarioBros-%d-%d-v0" % (i, j) for i in range(1, 9) for j in range(1, 5)]

    assert workers * parallel % len(games) == 0
    epsilon_nums = num_servers * workers * parallel // len(games)
    epsilons = np.power(epsilon, np.linspace(
        1, epsilon_power, epsilon_nums))
    epsilons = np.split(epsilons, num_servers)[server_id]

    tuples = []
    for i in games:
        for j in epsilons:
            tuples.append((i, j))

    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.015))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    phs = dict()

    phs["s"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, frames])
    phs["prev_a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 256 * 2])
    phs["slots"] = tf.placeholder(dtype=tf.float32, shape=[None, None])

    global_step = tf.train.get_or_create_global_step()

    with tf.device("/gpu"):
        lstm = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")
        model = Model(7, lstm, "agent", **phs)

    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=6)

    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    ckpt_path = None
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    context = zmq.Context()
    frontend = context.socket(zmq.ROUTER)
    frontend.bind(address)

    queue_ins = OrderedDict()
    # queue_out = Queue(maxsize=3 * tmplimit)
    for i in range(workers):
        queue_in = Queue()
        worker_id = i
        queue_ins[worker_id] = queue_in

        ts = tuples[i * parallel: (i + 1) * parallel]
        games = [t[0] for t in ts]
        epsilons = [t[1] for t in ts]

        worker = Process(
            target=Worker_Q,
            args=(queue_in,
                  address,
                  parallel,
                  BASE_DIR,
                  DATA_DIR,
                  3 * tmplimit,
                  server_id,
                  worker_id,
                  games,
                  frames,
                  n_step,
                  gamma,
                  seqlen,
                  burn_in,
                  epsilons))
        worker.daemon = True
        worker.start()

    while True:
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt is not None:
            new_ckpt_path = ckpt.model_checkpoint_path
            if new_ckpt_path != ckpt_path:
                ckpt_path = new_ckpt_path
                saver.restore(sess, ckpt_path)

        fd = dict()

        idx, msg = frontend.recv_multipart(copy=False)
        worker_id, databack = unpack(msg)
        s, a, state_in = databack
        fd[phs["s"]] = s
        fd[phs["prev_a"]] = a
        fd[phs["state_in"]] = state_in
        fd[phs["slots"]] = np.ones_like(a)

        _a_t_new, _state_out_batch = sess.run(
            [model.get_current_act(),
             model.state_out],
            feed_dict=fd)

        dataforward = (_a_t_new,
                       _state_out_batch)
        queue_ins[worker_id].put(dataforward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-SCRIPT_DIR", type=str,
                        default="/opt/tiger/test_ppo")
    parser.add_argument("-BASE_DIR", type=str,
                        default="/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
                                "/PPOcGAE_SuperMarioBros-v0/r2d3_2")
    parser.add_argument("-CKPT_DIR", type=str,
                        default="/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
                                "/PPOcGAE_SuperMarioBros-v0/r2d3_2/ckpt")
    parser.add_argument("-DATA_DIR", type=str,
                        default="/mnt/mytmpfs")
    parser.add_argument("-server_id", type=int, default=0)
    parser.add_argument("-num_servers", type=int, default=1)
    parser.add_argument("-frames", type=int, default=1)
    parser.add_argument("-workers", type=int, default=4)
    parser.add_argument("-worker_parallel", type=int, default=4)
    parser.add_argument("-n_step", type=int, default=5)
    parser.add_argument("-gamma", type=float, default=0.99)
    parser.add_argument("-seqlen", type=int, default=80)
    parser.add_argument("-burn_in", type=int, default=40)
    parser.add_argument("-epsilon", type=float, default=0.4)
    parser.add_argument("-epsilon_power", type=int, default=8)
    args = parser.parse_args()
    run(**args.__dict__)
    pass
