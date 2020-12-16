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

from XtoBeREMOVED.IMPALA_super_mario_bros_v1.Worker_PPOvtrace import Worker_Q
from XtoBeREMOVED.IMPALA_super_mario_bros_v1.Trainer_PPOvtrace import Model

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.getLogger('tensorflow').setLevel(logging.ERROR)


def clean(episodic, lifelong, data_dir, postfix):
    if lifelong is not None:
        if episodic >= lifelong:
            episodic -= lifelong
            sEPID = str(episodic)
            sEPID = (8 - len(sEPID)) * "0" + sEPID
            pattern = os.path.join(data_dir, sEPID + "_*." + postfix)
            names = glob.glob(pattern)
            for name in names:
                if os.path.exists(name):
                    try:
                        os.remove(name)
                    except FileNotFoundError:
                        pass


def run(**kwargs):
    tmplimit = 512
    lifelong = None

    server_id = kwargs.get("server_id", 0)

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
    MAX_STEPS = kwargs.get("max_steps", 3200)
    seqlen = kwargs.get("seqlen", 32)
    burn_in = kwargs.get("burn_in", 32)
    gamma = kwargs.get("gamma", 0.99)
    act_space = kwargs.get("act_space", 7)

    games = ["SuperMarioBros-1-1-v0",
             "SuperMarioBros-2-1-v0",
             "SuperMarioBros-4-1-v0",
             "SuperMarioBros-5-1-v0"]
    games = ["SuperMarioBros-%d-%d-v0" % (i, j) for i in range(1, 9) for j in range(1, 5)]

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

    with tf.device("/gpu"):
        lstm = tf.compat.v1.keras.layers.LSTM(
                256, return_sequences=True, return_state=True, name="lstm")
        model = Model(act_space, lstm, gamma, "agent", **phs)

    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=6)

    # while True:
    #     ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    #     if ckpt is not None:
    #         ckpt_path = ckpt.model_checkpoint_path
    #         if ckpt_path is not None:
    #             break
    #     sleep_time = 10
    #     logging.warning("No Model, Sleep %d seconds" % sleep_time)
    #     time.sleep(sleep_time)
    # saver.restore(sess, ckpt_path)

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
                  "\t".join(games),
                  frames,
                  seqlen,
                  burn_in,
                  act_space))
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
        fd[model.s_t] = s
        fd[model.previous_actions] = a
        fd[model.state_in] = state_in
        fd[model.slots] = np.ones_like(a)

        _a_t_new, _a_t_logits, _state_out_batch, _v_cur = sess.run(
            [model.get_current_act(),
             model.get_current_act_logits(),
             model.state_out,
             model.vf],
            feed_dict=fd)

        dataforward = (_a_t_new,
                       _a_t_logits,
                       _state_out_batch,
                       _v_cur)
        queue_ins[worker_id].put(dataforward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-SCRIPT_DIR", type=str,
                        default="/opt/tiger/test_ppo")
    parser.add_argument("-BASE_DIR", type=str,
                        default="/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
                                "/PPOcGAE_SuperMarioBros-v0/2")
    parser.add_argument("-CKPT_DIR", type=str,
                        default="/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
                                "/PPOcGAE_SuperMarioBros-v0/2/ckpt")
    parser.add_argument("-DATA_DIR", type=str,
                        default="/mnt/mytmpfs")
    parser.add_argument("-server_id", type=int, default=0)
    parser.add_argument("-frames", type=int, default=1)
    parser.add_argument("-workers", type=int, default=4)
    parser.add_argument("-worker_parallel", type=int, default=4)
    parser.add_argument("-max_steps", type=int, default=3200)
    parser.add_argument("-seqlen", type=int, default=32)
    parser.add_argument("-burn_in", type=int, default=32)
    parser.add_argument("-gamma", type=float, default=0.99)
    parser.add_argument("-act_space", type=int, default=7)
    args = parser.parse_args()
    run(**args.__dict__)
    pass
