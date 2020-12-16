# coding: utf-8

import sys

sys.path.append("/opt/tiger/test_ppo")

import tensorflow as tf
import numpy as np
import os
import copy
import logging
from collections import namedtuple
import time
import glob
import pickle
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool, Process, Queue
from threading import Thread
import pyarrow as pa

from utils import get_shape
from utils import unpack
from algorithm import dPPOcC
from algorithm import dAC
from infer import categorical
from module import entropy_from_logits as entropy
from module import vtrace_from_logits
from train_ops import miniOp

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.getLogger('tensorflow').setLevel(logging.ERROR)

nest = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train_ops", "mode")
flags.DEFINE_integer("act_space", 12, "act space")

flags.DEFINE_string(
    "basedir", "/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
               "/PPOcGAE_SuperMarioBros-v0", "base dir")
flags.DEFINE_string("dir", "vtrace0", "dir number")
flags.DEFINE_string(
    "datadir", "/mnt/mytmpfs", "data dir")
flags.DEFINE_string(
    "scriptdir", "/opt/tiger/test_ppo/examples/IMPALA_super_mario_bros_v1", "script dir")

flags.DEFINE_bool("use_stage", True, "whether to use tf.contrib.staging")

flags.DEFINE_integer("num_servers", 8, "number of servers")
flags.DEFINE_integer("num_workers", 4, "number of workers")
flags.DEFINE_integer("worker_parallel", 64, "parallel workers")
flags.DEFINE_integer("max_steps", 3200, "max rollout steps")
flags.DEFINE_integer("seqlen", 32, "seqlen of each training segment")
flags.DEFINE_integer("burn_in", 32, "seqlen of each burn-in segment")
flags.DEFINE_integer("batch_size", 192, "batch size")
flags.DEFINE_integer("total_environment_frames", 1000000000,
                     "total num of frames for train")
flags.DEFINE_integer("buffer_size", 300000, "num of frames in buffer")
flags.DEFINE_integer("num_replay", 1, "num of replays on avg")

flags.DEFINE_float("init_lr", 1e-3, "initial learning rate")
flags.DEFINE_float("lr_decay", 1.0, "whether to decay learning rate")
flags.DEFINE_string("opt", "adam", "which optimizer to use")
flags.DEFINE_float("warmup_steps", 4000, "steps for warmup")

flags.DEFINE_integer("frames", 1, "stack of frames for each state")
flags.DEFINE_integer("image_size", 84, "input image size")
flags.DEFINE_float("vf_clip", 1.0, "clip of value function")
flags.DEFINE_float("ppo_clip", 0.2, "clip of ppo loss")
flags.DEFINE_float("gamma", 0.99, "discount rate")
flags.DEFINE_float("pi_coef", 20.0, "weight of policy fn loss")
flags.DEFINE_float("vf_coef", 1.0, "weight of value fn loss")
flags.DEFINE_float("ent_coef", 1.0, "weight of entropy loss")
flags.DEFINE_bool("zero_init", False, "whether to zero init initial state")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")
flags.DEFINE_bool("normalize_advantage", False,
                  "whether to normalize advantage")

flags.DEFINE_integer("seed", 12358, "random seed")


class Model(object):
    def __init__(self,
                 act_space,
                 lstm,
                 gamma,
                 scope="agent",
                 **kwargs):
        self.act_space = act_space
        self.scope = scope

        self.s_t = kwargs.get("s")
        self.previous_actions = kwargs.get("prev_a")
        self.state_in = kwargs.get("state_in")
        self.slots = tf.cast(kwargs.get("slots"), tf.float32)

        prev_a = tf.one_hot(
            self.previous_actions, depth=act_space, dtype=tf.float32)

        s_feature, self.state_out = self.feature_net(
            self.s_t, lstm, prev_a, self.state_in, scope + "_feature")

        self.current_act_logits = self.a_net(
            s_feature, scope + "_a")
        self.current_act = tf.squeeze(
            categorical(self.current_act_logits), axis=-1)

        self.vf = self.v_net(
            s_feature,
            scope + "_value") * self.slots

        self.bootstrap_s = kwargs.get("bootstrap_s")
        if self.bootstrap_s is not None:
            self.bootstrap_slots = tf.cast(kwargs.get("bootstrap_slots"), tf.float32)
            self.r_t = kwargs.get("r")
            self.old_vf = kwargs.get("v_cur")
            self.old_act_logits = kwargs.get("a_logits")
            self.a_t = kwargs.get("a")
            a_onehot = tf.one_hot(
                self.a_t, depth=act_space, dtype=tf.float32)

            bootstrap_feature, _ = self.feature_net(
                self.bootstrap_s[:, None, :, :, :],
                lstm, a_onehot[:, -2:-1, :],
                self.state_out, scope + "_feature")

            bootstrap_feature = bootstrap_feature[:, -1, :]
            bootstrap_value = self.v_net(
                bootstrap_feature,
                scope + "_value") * self.bootstrap_slots

            vtrace = vtrace_from_logits(
                self.old_act_logits, self.current_act_logits,
                self.a_t, gamma * tf.ones_like(self.a_t, tf.float32),
                self.r_t, self.vf, bootstrap_value)

            self.vs = vtrace.vs
            self.adv = vtrace.advantages
            self.pg_adv = vtrace.pg_advantages

    def get_current_act(self):
        return self.current_act

    def get_current_act_logits(self):
        return self.current_act_logits

    def v_net(self, feature, scope):
        net = feature
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # net = tf.layers.dense(
            #     net,
            #     get_shape(feature)[-1],
            #     activation=tf.nn.relu,
            #     name="dense")
            v_value = tf.squeeze(
                tf.layers.dense(
                    net,
                    1,
                    activation=None,
                    name="v_value"),
                axis=-1)

        return v_value

    def a_net(self, feature, scope):
        net = feature
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # net = tf.layers.dense(
            #     net,
            #     get_shape(feature)[-1],
            #     activation=tf.nn.relu,
            #     name="dense")
            act_logits = tf.layers.dense(
                net,
                self.act_space,
                activation=None,
                name="a_logits")

        return act_logits

    def feature_net(self, image, lstm, prev_a, state_in, scope="feature"):
        shape = get_shape(image)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            image = tf.reshape(image, [-1] + shape[-3:])
            filter = [16, 32, 32]
            kernel = [(3, 3), (3, 3), (5, 3)]
            stride = [(1, 2), (1, 2), (2, 1)]
            for i in range(len(filter)):
                image = tf.layers.conv2d(
                    image,
                    filters=filter[i],
                    kernel_size=kernel[i][0],
                    strides=stride[i][0],
                    padding="valid",
                    activation=None,
                    name="conv_%d" % i)
                image = tf.layers.max_pooling2d(
                    image,
                    pool_size=kernel[i][1],
                    strides=stride[i][1],
                    padding="valid",
                    name="maxpool_%d" % i)
                image = self.resblock(
                    image, "res0_%d" % i)
                # image = self.resblock(
                #     image, "res1_%d" % i)
            image = tf.nn.relu(image)

            new_shape = get_shape(image)
            feature = tf.reshape(
                image, [shape[0], shape[1], new_shape[1] * new_shape[2] * new_shape[3]])

            feature = tf.layers.dense(feature, 256, tf.nn.relu, name="feature")
            feature = tf.concat([feature, prev_a], axis=-1)
            c_in, h_in = tf.split(state_in, 2, axis=-1)
            feature, c_out, h_out = lstm(
                feature, initial_state=[c_in, h_in])
            state_out = tf.concat([c_in, h_in], axis=-1)
        return feature, state_out

    @staticmethod
    def resblock(tensor, scope):
        shape = get_shape(tensor)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            res = tf.nn.relu(tensor)
            res = tf.layers.conv2d(
                res,
                filters=shape[-1],
                kernel_size=3,
                strides=1,
                padding="same",
                activation=None,
                name="conv0")
            res = tf.nn.relu(res)
            res = tf.layers.conv2d(
                res,
                filters=shape[-1],
                kernel_size=3,
                strides=1,
                padding="same",
                activation=None,
                name="conv1")
            output = res + tensor
        return output


def build_learner(pre, post, act_space, num_frames):
    global_step = tf.train.get_or_create_global_step()
    init_lr = FLAGS.init_lr
    decay = FLAGS.lr_decay
    warmup_steps = FLAGS.warmup_steps

    global_step_float = tf.cast(global_step, tf.float32)

    lr = tf.train.polynomial_decay(
        init_lr, global_step,
        FLAGS.total_environment_frames * FLAGS.num_replay // (
                FLAGS.batch_size * FLAGS.seqlen // 2),
        init_lr / 10.)
    is_warmup = tf.cast(global_step_float < warmup_steps, tf.float32)
    lr = is_warmup * global_step_float / warmup_steps * init_lr + (
            1.0 - is_warmup) * (init_lr * (1.0 - decay) + lr * decay)
    if FLAGS.opt == "adam":
        optimizer = tf.train.AdamOptimizer(lr)
    else:
        optimizer = tf.train.RMSPropOptimizer(lr, epsilon=0.01)

    ent_coef = tf.train.polynomial_decay(
        FLAGS.ent_coef, global_step,
        FLAGS.total_environment_frames * FLAGS.num_replay // (
                FLAGS.batch_size * FLAGS.seqlen // 2),
        FLAGS.ent_coef / 10.)

    if FLAGS.zero_init:
        pre["state_in"] = tf.zeros_like(pre["state_in"])
        pre["state1_in"] = tf.zeros_like(pre["state1_in"])

    lstm = tf.compat.v1.keras.layers.LSTM(
        256, return_sequences=True, return_state=True, name="lstm")
    pre_model = Model(act_space, lstm, FLAGS.gamma, "agent", **pre)

    post["state_in"] = tf.stop_gradient(pre_model.state_out)

    post_model = Model(act_space, lstm, FLAGS.gamma, "agent", **post)

    if FLAGS.ppo_clip > 0.0:
        advantage = post_model.adv
        adv_mean = tf.reduce_mean(advantage)
        tf.summary.scalar("adv_mean", adv_mean)
        advantage_center = advantage - adv_mean
        adv_std = tf.sqrt(tf.reduce_mean(advantage_center ** 2))
        tf.summary.scalar("adv_std", adv_std)
        if FLAGS.normalize_advantage:
            advantage = advantage_center / tf.maximum(adv_std, 1e-12)

        losses = dPPOcC(post_model.a_t,
                        post_model.current_act_logits,
                        post_model.old_act_logits,
                        advantage,
                        FLAGS.ppo_clip,
                        post_model.vf,
                        post_model.vs,
                        FLAGS.vf_clip,
                        post_model.old_vf)
    else:
        advantage = post_model.pg_adv
        adv_mean = tf.reduce_mean(advantage)
        tf.summary.scalar("adv_mean", adv_mean)
        advantage_center = advantage - adv_mean
        adv_std = tf.sqrt(tf.reduce_mean(advantage_center ** 2))
        tf.summary.scalar("adv_std", adv_std)
        if FLAGS.normalize_advantage:
            advantage = advantage_center / tf.maximum(adv_std, 1e-12)

        losses = dAC(post_model.a_t,
                     post_model.current_act_logits,
                     advantage,
                     post_model.vf,
                     post_model.vs,
                     FLAGS.vf_clip,
                     post_model.old_vf)
    entropy_loss = tf.reduce_mean(
        entropy(post_model.current_act_logits) * post_model.slots)

    p_loss = tf.reduce_mean(losses.p_loss * post_model.slots)
    v_loss = tf.reduce_mean(losses.v_loss * post_model.slots)

    loss = FLAGS.pi_coef * p_loss \
           + FLAGS.vf_coef * v_loss \
           - ent_coef * entropy_loss
    train_op = miniOp(optimizer, loss, FLAGS.grad_clip)

    new_frames = tf.reduce_sum(post["slots"])
    num_frames_and_train = tf.assign_add(
        num_frames, new_frames)

    with tf.control_dependencies([train_op]):
        global_step_and_train = tf.assign_add(
            global_step, 1)

    tf.summary.scalar("learning_rate", lr)
    tf.summary.scalar("ent_coef", ent_coef)
    tf.summary.scalar("ent_loss", entropy_loss)
    tf.summary.scalar("p_loss", p_loss)
    tf.summary.scalar("v_loss", v_loss)
    tf.summary.scalar("all_loss", loss)

    return num_frames_and_train, global_step_and_train


class QueueReader(Thread):
    def __init__(self,
                 sess,
                 global_queue,
                 pattern,
                 keys,
                 dtypes,
                 shapes):
        Thread.__init__(self)
        self.daemon = True

        self.sess = sess
        self.global_queue = global_queue
        self.pattern = pattern

        self.keys = keys
        self.placeholders = [
            tf.placeholder(
                dtype=dtype, shape=shape
            ) for dtype, shape in zip(dtypes, shapes)]
        self.enqueue_op = self.global_queue.enqueue(
            dict(zip(keys, self.placeholders)))
        self.generator = self.next()

        self.count = 0
        self.retime = 0
        self.untime = 0

    @staticmethod
    def read(name):
        try:
            start_time = time.time()
            with pa.OSFile(name) as f:
                s = f.read_buffer()
            readtime = time.time() - start_time
            start_time = time.time()
            s = unpack(s)
            untime = time.time() - start_time
            return s, readtime, untime
        except Exception as e:
            logging.warning(e)
            return None

    def enqueue(self):
        seg = next(self.generator)
        fd = {self.placeholders[i]: seg[key] for i, key in enumerate(self.keys)}
        self.sess.run(self.enqueue_op, fd)

    def next(self):
        while True:
            names = glob.glob(self.pattern)
            if not names:
                time.sleep(1)
                continue
            np.random.shuffle(names)
            while names:
                name = names.pop()
                seg = self.read(name)
                if seg is not None:
                    if os.path.exists(name):
                        os.remove(name)
                    if os.path.exists(name[:-9] + ".log"):
                        os.remove(name[:-9] + ".log")
                    seg, retime, untime = seg
                    # while self.global_queue.qsize() >= self.max_size:
                    #     time.sleep(1)
                    # self.global_queue.enqueue(seg)
                    self.count += 1
                    self.retime += retime
                    self.untime += untime
                    if self.count % 100 == 0:
                        # logging.info(
                        #     "Read time %.2f, Unpack time %.2f"
                        #     % (self.retime, self.untime))
                        self.count = 0
                        self.retime = 0
                        self.untime = 0
                    yield seg

    def run(self):
        while True:
            self.enqueue()


def train():
    act_space = FLAGS.act_space
    BASE_DIR = os.path.join(FLAGS.basedir, FLAGS.dir)
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    CKPT_DIR = os.path.join(BASE_DIR, "ckpt")
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)
    DATA_DIR = FLAGS.datadir
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    SCRIPT_DIR = FLAGS.scriptdir

    for i in range(FLAGS.num_servers):
        os.system("python3 %s/Server_PPOvtrace.py "
                  "-server_id %d "
                  "-SCRIPT_DIR %s "
                  "-BASE_DIR %s "
                  "-CKPT_DIR %s "
                  "-DATA_DIR %s "
                  "-frames %d "
                  "-workers %d "
                  "-worker_parallel %d "
                  "-max_steps %d "
                  "-seqlen %d "
                  "-burn_in %d "
                  "-gamma %.6f "
                  "-act_space %d "
                  "&" % (
                      SCRIPT_DIR,
                      i,
                      SCRIPT_DIR,
                      BASE_DIR,
                      CKPT_DIR,
                      DATA_DIR,
                      FLAGS.frames,
                      FLAGS.num_workers,
                      FLAGS.worker_parallel,
                      FLAGS.max_steps,
                      FLAGS.seqlen,
                      FLAGS.burn_in,
                      FLAGS.gamma,
                      act_space))

    logging.basicConfig(filename=os.path.join(
        BASE_DIR, "Trainerlog"), level="INFO")

    tf.set_random_seed(FLAGS.seed)

    num_frames = tf.get_variable(
        'num_environment_frames',
        initializer=tf.zeros_initializer(),
        shape=[],
        dtype=tf.int32,
        trainable=False)
    tf.summary.scalar("frames", num_frames)
    global_step = tf.train.get_or_create_global_step()

    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.94))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    structure = None
    while True:
        names = glob.glob(os.path.join(DATA_DIR, "*.seg"))
        while names:
            name = names.pop()
            seg = QueueReader.read(name)
            if seg is not None:
                structure = seg[0]
                flatten_structure = nest.flatten(structure)
                break
        if structure is not None:
            break
        logging.warning("NO DATA, SLEEP %d seconds" % 60)
        time.sleep(60)

    keys = list(structure.keys())
    dtypes = [structure[k].dtype for k in keys]
    shapes = [structure[k].shape for k in keys]
    segBuffer = tf.queue.FIFOQueue(
        capacity=4 * FLAGS.batch_size,
        dtypes=dtypes,
        shapes=shapes,
        names=keys,
        shared_name="buffer")
    replayBuffer = tf.queue.RandomShuffleQueue(
        capacity=int(1.2 * FLAGS.buffer_size // (FLAGS.seqlen + FLAGS.burn_in)),
        min_after_dequeue=int(0.8 * FLAGS.buffer_size // (FLAGS.seqlen + FLAGS.burn_in)),
        dtypes=dtypes,
        shapes=shapes,
        names=keys,
        shared_name="repbuffer")

    readers = []
    patterns = [os.path.join(
        DATA_DIR, "*_%s_*_*.seg" % ((4 - len(str(i))) * "0" + str(i))
    ) for i in range(FLAGS.num_workers)]
    for pattern in patterns:
        reader = QueueReader(
            sess=sess,
            global_queue=segBuffer,
            pattern=pattern,
            keys=keys,
            dtypes=dtypes,
            shapes=shapes)
        reader.start()
        readers.append(reader)

    def get_phs(dequeued):
        prephs, postphs = dict(), dict()
        for k, v in dequeued.items():
            if k == "state_in":
                prephs[k] = v
            elif k == "bootstrap_s" or k == "bootstrap_slots":
                postphs[k] = v
            else:
                prephs[k], postphs[k] = tf.split(
                    v, [FLAGS.burn_in, FLAGS.seqlen], axis=1)
        return prephs, postphs

    dequeued1 = segBuffer.dequeue_many(FLAGS.batch_size)
    enqueue_op1 = replayBuffer.enqueue_many(dequeued1)
    prephs1, postphs1 = get_phs(dequeued1)

    dequeued2 = replayBuffer.dequeue_many(FLAGS.batch_size)
    enqueue_op2 = replayBuffer.enqueue_many(dequeued2)
    prephs2, postphs2 = get_phs(dequeued2)

    with tf.device("/gpu"):
        if FLAGS.use_stage:
            area = tf.contrib.staging.StagingArea(
                [prephs1[key].dtype for key in prephs1] + [postphs1[key].dtype for key in postphs1],
                [prephs1[key].shape for key in prephs1] + [postphs1[key].shape for key in postphs1])
            if FLAGS.num_replay > 1:
                with tf.control_dependencies([enqueue_op1]):
                    stage_op1 = area.put([prephs1[key] for key in prephs1] + [postphs1[key] for key in postphs1])
            else:
                stage_op1 = area.put([prephs1[key] for key in prephs1] + [postphs1[key] for key in postphs1])
            with tf.control_dependencies([enqueue_op2]):
                stage_op2 = area.put([prephs2[key] for key in prephs2] + [postphs2[key] for key in postphs2])
            stage_op3 = area.put([prephs2[key] for key in prephs2] + [postphs2[key] for key in postphs2])
            from_stage = area.get()
            predatas = {key: from_stage[i] for i, key in enumerate(prephs1)}
            postdatas = {key: from_stage[i + len(prephs1)] for i, key in enumerate(postphs1)}
        else:
            stage_op1 = stage_op2 = stage_op3 = []
            predatas, postdatas = prephs1, postphs1

        num_frames_and_train, global_step_and_train = build_learner(
            pre=predatas, post=postdatas, act_space=act_space, num_frames=num_frames)

    summary_ops = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(os.path.join(BASE_DIR, "summary"), sess.graph)

    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=6)

    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    saver.save(sess, os.path.join(CKPT_DIR, "PPOcGAE"), global_step=global_step)

    total_frames = 0
    sess.run(stage_op1)

    def step(num_frames_and_train, total_frames, other_ops):
        if num_frames_and_train is None:
            gs, summary, _ = sess.run(other_ops)
        else:
            total_frames, gs, summary, _ = sess.run(
                [num_frames_and_train] + other_ops)
        summary_writer.add_summary(summary, global_step=gs)

        msg = "  Global Step %d, Total Frames %d, Time Consume %.2f" % (
            gs, total_frames, time.time() - start)
        logging.info(msg)
        if gs % 25 == 0:
            saver.save(
                sess, os.path.join(CKPT_DIR, "PPOcGAE"),
                global_step=global_step)
        return total_frames

    for i in range(FLAGS.buffer_size // FLAGS.batch_size // FLAGS.seqlen):
        start = time.time()

        total_frames = step(
            num_frames_and_train, total_frames,
            [global_step_and_train, summary_ops, stage_op1])

    while total_frames < FLAGS.total_environment_frames:
        start = time.time()

        for i in range(FLAGS.num_replay):
            if i == 0:
                total_frames = step(
                    num_frames_and_train, total_frames,
                    [global_step_and_train, summary_ops, stage_op1])
            elif i < FLAGS.num_replay - 1:
                total_frames = step(
                    None, total_frames,
                    [global_step_and_train, summary_ops, stage_op2])
            else:
                total_frames = step(
                    None, total_frames,
                    [global_step_and_train, summary_ops, stage_op3])


def main(_):
    if FLAGS.mode == "train":
        train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
