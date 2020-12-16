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
from queue import PriorityQueue
from threading import Thread
import pyarrow as pa

from utils import get_shape
from utils import unpack
from infer import categorical
from module import duelingQ
from module import doubleQ
from module import mse
from module import rescaleTarget
from train_ops import miniOp
from train_ops import assignOp

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.getLogger('tensorflow').setLevel(logging.ERROR)

nest = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")
flags.DEFINE_integer("act_space", 12, "act space")

flags.DEFINE_string(
    "basedir", "/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
               "/PPOcGAE_SuperMarioBros-v0", "base dir")
flags.DEFINE_string("dir", "r2d2_2", "dir number")
flags.DEFINE_string(
    "datadir", "/mnt/mytmpfs", "data dir")
flags.DEFINE_string(
    "scriptdir", "/opt/tiger/test_ppo/zR2D2", "script dir")

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
flags.DEFINE_integer("num_replay", 4, "num of replays on avg")

flags.DEFINE_float("init_lr", 1e-3, "initial learning rate")
flags.DEFINE_float("lr_decay", 1.0, "whether to decay learning rate")
flags.DEFINE_string("opt", "adam", "which optimizer to use")
flags.DEFINE_float("warmup_steps", 4000, "steps for warmup")

flags.DEFINE_integer("frames", 1, "stack of frames for each state")
flags.DEFINE_integer("image_size", 84, "input image size")
flags.DEFINE_float("epsilon", 0.4, "random exploration rate")
flags.DEFINE_integer("epsilon_power", 8, "max power of random exploration rate")
flags.DEFINE_float("gamma", 0.99, "discount rate")
flags.DEFINE_float("qf_clip", 1.0, "clip of value function")
flags.DEFINE_integer("n_step", 5, "n_step td error")
flags.DEFINE_integer("target_update", 2500, "target net update per steps")
flags.DEFINE_bool("smooth_update", False, "whether to smooth update target net")
flags.DEFINE_bool("rescale", True, "whether to rescale target")
flags.DEFINE_bool("zero_init", True, "whether to zero init initial state")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")

flags.DEFINE_integer("seed", 12358, "random seed")


class Model(object):
    def __init__(self,
                 act_space,
                 lstm,
                 scope="agent",
                 **kwargs):
        self.act_space = act_space
        self.scope = scope

        self.s = kwargs.get("s")
        self.prev_a = kwargs.get("prev_a")
        self.state_in = kwargs.get("state_in")
        self.slots = tf.cast(kwargs.get("slots"), tf.float32)

        feature, self.state_out = self.feature_net(
            self.s, lstm, self.prev_a, self.state_in)

        self.qf = self.q_fn(
            feature, self.slots, self.scope + "_current")

        self.current_act = tf.argmax(self.qf, axis=-1)

        # base_probs = tf.ones(
        #     get_shape(self.prev_a) + [act_space]
        # ) * epsilon / tf.cast(act_space, tf.float32)
        # argmax_a = tf.argmax(self.qf, axis=-1)
        # argmax_probs = tf.one_hot(
        #     argmax_a, depth=act_space, dtype=tf.float32
        # ) * (1.0 - epsilon)
        # self.current_act_probs = base_probs + argmax_probs
        #
        # self.current_act = tf.squeeze(
        #     categorical(tf.math.log(self.current_act_probs)), axis=-1)

        self.bootstrap_s = kwargs.get("bootstrap_s")
        if self.bootstrap_s is not None:
            self.bootstrap_prev_a = kwargs.get("bootstrap_prev_a")
            self.bootstrap_slots = tf.cast(
                kwargs.get("bootstrap_slots"), tf.float32)
            self.a = kwargs.get("a")
            self.r = kwargs.get("r")

            self.qa = tf.reduce_sum(
                tf.one_hot(
                    self.a, depth=self.act_space, dtype=tf.float32
                ) * self.qf, axis=-1)

            bootstrap_feature, _ = self.feature_net(
                self.bootstrap_s, lstm, self.bootstrap_prev_a, self.state_out)

            n_step = get_shape(bootstrap_feature)[1]

            feature1 = tf.concat(
                [feature[:, n_step:, :], bootstrap_feature], axis=1)
            slots1 = tf.concat(
                [self.slots[:, n_step:], self.bootstrap_slots], axis=1)
            self.q1f1 = self.q_fn(
                feature1, slots1, self.scope + "_target")

            self.q1f = self.q_fn(
                feature1, slots1, self.scope + "_current")

            self.qa1 = doubleQ(self.q1f1, self.q1f)

    def get_current_act(self):
        return self.current_act

    def q_fn(self, feature, slots, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            vf = self.v_net(feature)
            adv = self.adv_net(feature)
            qf = duelingQ(vf, adv)
            qf = qf * tf.expand_dims(slots, axis=-1)
        return qf

    def v_net(self, feature, scope="v_net"):
        net = feature
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(
                net,
                get_shape(feature)[-1],
                activation=tf.nn.relu,
                name="dense")
            v_value = tf.squeeze(
                tf.layers.dense(
                    net,
                    1,
                    activation=None,
                    name="v_value"),
                axis=-1)
        return v_value

    def adv_net(self, feature, scope="adv_net"):
        net = feature
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(
                net,
                get_shape(feature)[-1],
                activation=tf.nn.relu,
                name="dense")
            adv = tf.layers.dense(
                net,
                self.act_space,
                activation=None,
                name="q_value")
        return adv

    def feature_net(self, image, lstm, a_t0, state_in, scope="feature"):
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

            prev_a = tf.one_hot(
                a_t0, depth=self.act_space, dtype=tf.float32)
            c_in, h_in = tf.split(state_in, 2, axis=-1)

            feature = tf.layers.dense(feature, 256, tf.nn.relu, name="feature")
            feature = tf.concat([feature, prev_a], axis=-1)

            feature, c_out, h_out = lstm(
                feature, initial_state=[c_in, h_in])
            state_out = tf.concat([c_out, h_out], axis=-1)

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
                FLAGS.batch_size * FLAGS.seqlen),
        init_lr / 10.)
    is_warmup = tf.cast(global_step_float < warmup_steps, tf.float32)
    lr = is_warmup * global_step_float / warmup_steps * init_lr + (
            1.0 - is_warmup) * (init_lr * (1.0 - decay) + lr * decay)

    if FLAGS.opt == "adam":
        optimizer = tf.train.AdamOptimizer(lr)
    else:
        optimizer = tf.train.RMSPropOptimizer(lr, epsilon=0.01)

    if FLAGS.zero_init:
        pre["state_in"] = tf.zeros_like(pre["state_in"])

    lstm = tf.compat.v1.keras.layers.LSTM(
        256, return_sequences=True, return_state=True, name="lstm")
    pre_model = Model(act_space, lstm, "agent", **pre)

    post["state_in"] = tf.stop_gradient(pre_model.state_out)

    post_model = Model(act_space, lstm, "agent", **post)

    if FLAGS.rescale:
        target = rescaleTarget(
            post_model.r, FLAGS.gamma ** FLAGS.n_step, post_model.qa1)
    else:
        target = post_model.r + FLAGS.gamma ** FLAGS.n_step * post_model.qa1

    loss = 100. * tf.reduce_mean(
        post_model.slots * mse(
            post_model.qa,
            tf.stop_gradient(target)))

    exp_td = post_model.slots * tf.math.pow(
        tf.abs(post_model.qa - (
                post_model.r + FLAGS.gamma **
                FLAGS.n_step * post_model.qa1)), 0.9)

    avg_p = tf.reduce_sum(exp_td, axis=-1) / (
        tf.reduce_sum(post_model.slots, axis=-1))
    max_p = tf.reduce_max(exp_td, axis=-1)

    priority = 0.9 * max_p + 0.1 * avg_p
    priority = tf.cast(-10000 * priority, tf.int64)

    train_op = miniOp(optimizer, loss, FLAGS.grad_clip)

    init_target_op = assignOp(
        1.0,
        {post_model.scope + "_current":
             post_model.scope + "_target"})
    if FLAGS.smooth_update:
        assign_op = assignOp(
            1.0 / FLAGS.target_update,
            {post_model.scope + "_current":
                 post_model.scope + "_target"})
        dependency = [train_op, assign_op]
    else:
        dependency = [train_op]

    new_frames = tf.reduce_sum(post["slots"])

    with tf.control_dependencies(dependency):
        global_step_and_train = tf.assign_add(
            global_step, 1)
        num_frames_and_train = tf.assign_add(
            num_frames, new_frames)

    tf.summary.scalar("learning_rate", lr)
    tf.summary.scalar("all_loss", loss)

    return (num_frames_and_train,
            global_step_and_train,
            init_target_op,
            priority)


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
        os.system("python3 %s/Server_r2d2.py "
                  "-server_id %d "
                  "-num_servers %d "
                  "-SCRIPT_DIR %s "
                  "-BASE_DIR %s "
                  "-CKPT_DIR %s "
                  "-DATA_DIR %s "
                  "-frames %d "
                  "-workers %d "
                  "-worker_parallel %d "
                  "-n_step %d "
                  "-seqlen %d "
                  "-burn_in %d "
                  "-gamma %.6f "
                  "-epsilon %.6f "
                  "-epsilon_power %d "
                  "-act_space %d "
                  "&" % (
                      SCRIPT_DIR,
                      i,
                      FLAGS.num_servers,
                      SCRIPT_DIR,
                      BASE_DIR,
                      CKPT_DIR,
                      DATA_DIR,
                      FLAGS.frames,
                      FLAGS.num_workers,
                      FLAGS.worker_parallel,
                      FLAGS.n_step,
                      FLAGS.seqlen,
                      FLAGS.burn_in,
                      FLAGS.gamma,
                      FLAGS.epsilon,
                      FLAGS.epsilon_power,
                      FLAGS.act_space
                  ))

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
            per_process_gpu_memory_fraction=1.0))
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
    replayBuffer = tf.queue.PriorityQueue(
        capacity=int(1.2 * FLAGS.buffer_size // (FLAGS.seqlen + FLAGS.burn_in)),
        types=dtypes,
        shapes=shapes,
        names=["priority"] + keys,
        shared_name="priorityBuffer")

    with tf.name_scope("buffer_size"):
        tf.summary.scalar("segBuffer_size", segBuffer.size())
        tf.summary.scalar("replayBuffer_size", replayBuffer.size())

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
            elif k == "bootstrap_s" or k == "bootstrap_prev_a" or k == "bootstrap_slots":
                postphs[k] = v
            else:
                prephs[k], postphs[k] = tf.split(
                    v, [FLAGS.burn_in, FLAGS.seqlen], axis=1)
        return prephs, postphs

    replayBufferSize = replayBuffer.size()
    min_capacity = FLAGS.buffer_size // (FLAGS.seqlen + FLAGS.burn_in)
    dequeued = tf.cond(
        replayBufferSize < min_capacity,
        lambda: segBuffer.dequeue_many(FLAGS.batch_size),
        lambda: {k: v for k, v in replayBuffer.dequeue_many(
            FLAGS.batch_size).items() if k != "priority"})
    prephs, postphs = get_phs(dequeued)

    pph = tf.placeholder(dtype=tf.int64, shape=[None])
    phs = [
        tf.placeholder(
            dtype=dtype, shape=[None] + list(shape)
        ) for dtype, shape in zip(dtypes, shapes)]
    dphs = dict(zip(["priority"] + keys, [pph] + phs))
    enqueue_op = replayBuffer.enqueue_many(dphs)

    with tf.device("/gpu"):
        if FLAGS.use_stage:
            area = tf.contrib.staging.StagingArea(
                [prephs[key].dtype for key in prephs] + [postphs[key].dtype for key in postphs],
                [prephs[key].shape for key in prephs] + [postphs[key].shape for key in postphs])
            stage_op = area.put([prephs[key] for key in prephs] + [postphs[key] for key in postphs])
            from_stage = area.get()
            predatas = {key: from_stage[i] for i, key in enumerate(prephs)}
            postdatas = {key: from_stage[i + len(prephs)] for i, key in enumerate(postphs)}
        else:
            stage_op = []
            predatas, postdatas = prephs, postphs

        num_frames_and_train, global_step_and_train, assign_op, priority = build_learner(
            pre=predatas, post=postdatas, act_space=act_space, num_frames=num_frames)

    summary_ops = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(os.path.join(BASE_DIR, "summary"), sess.graph)

    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=6)

    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    saver.save(sess, os.path.join(CKPT_DIR, "CKPT"), global_step=global_step)

    total_frames = 0
    sess.run([stage_op, assign_op])
    fd = OrderedDict()

    while total_frames < FLAGS.total_environment_frames * FLAGS.num_replay:
        start = time.time()

        total_frames, gs, summary, _, size, d, p = sess.run(
            [num_frames_and_train, global_step_and_train,
             summary_ops, stage_op, replayBufferSize,
             dequeued, priority])

        if len(fd):
            if gs % FLAGS.num_replay == 0:
                n_p = fd[dphs["priority"]]
                sorted_np = sorted(n_p)
                threshold_id = len(sorted_np) * (
                        FLAGS.num_replay - 1) // FLAGS.num_replay
                threshold = sorted_np[threshold_id]
                idx = np.where(n_p < threshold)
                for key in dphs:
                    fd[dphs[key]] = fd[dphs[key]][idx]
                sess.run(enqueue_op,
                         feed_dict=fd)

                fd[dphs["priority"]] = p
                for key in keys:
                    fd[dphs[key]] = d[key]
            else:
                fd[dphs["priority"]] = np.concatenate(
                    (fd[dphs["priority"]], p), axis=0)
                for key in keys:
                    fd[dphs[key]] = np.concatenate(
                        (fd[dphs[key]], d[key]), axis=0)
        else:
            fd[dphs["priority"]] = p
            for key in keys:
                fd[dphs[key]] = d[key]

        summary_writer.add_summary(summary, global_step=gs)

        msg = "  Global Step %d, Total Frames %d, Time Consume %.2f" % (
            gs, total_frames, time.time() - start)
        logging.info(msg)
        if gs % 25 == 0:
            saver.save(
                sess, os.path.join(CKPT_DIR, "CKPT"),
                global_step=global_step)
        if gs % FLAGS.target_update == 0 and not FLAGS.smooth_update:
            sess.run(assign_op)


def main(_):
    if FLAGS.mode == "train":
        train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
