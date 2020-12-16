# coding: utf-8

import tensorflow as tf
import numpy as np
import random
import os
import logging
from collections import namedtuple
import time
import argparse
import glob
import pyarrow as pa
from PIL import Image

from utils import get_shape
from utils import get_gaes
from utils import pack

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

os.environ["CUDA_VISIBLE_DEVICES"] = ""
logging.getLogger('tensorflow').setLevel(logging.ERROR)
Seg = namedtuple("Seg", ["s", "a", "a_logits", "prev_a", "r", "s1", "gaes", "ret", "v_cur", "v_tar", "c_in", "h_in"])


class Model(object):
    def __init__(self,
                 act_space,
                 vf_clip,
                 lstm,
                 scope="agent",
                 **kwargs):
        self.act_space = act_space
        self.scope = scope

        self.s_t = kwargs.get("s")
        self.previous_actions = kwargs.get("prev_a")
        self.a_t = kwargs.get("a")
        self.old_act_logits = kwargs.get("a_logits")
        self.s_t1 = kwargs.get("s1")

        self.ret = kwargs.get("ret")
        self.advantage = kwargs.get("adv")
        self.target_value = kwargs.get("v_tar")
        self.old_current_value = kwargs.get("v_cur")

        self.c_in = kwargs.get("c_in")
        self.h_in = kwargs.get("h_in")

        self.slots = kwargs.get("slots")

        prev_a = tf.one_hot(
            self.previous_actions, depth=act_space + 1, dtype=tf.float32)

        s_current_feature, self.c_out, self.h_out = self.feature_net(
            self.s_t, lstm, prev_a, scope + "_current_feature")

        self.current_act_logits = self.a_net(
            s_current_feature, scope + "_acurrent")
        self.current_act = tf.squeeze(
            self.categorical(self.current_act_logits, 1), axis=-1)
        self.current_act_probs = tf.nn.softmax(self.current_act_logits, axis=-1)

        self.old_act = tf.squeeze(
            self.categorical(self.old_act_logits, 1), axis=-1)
        self.old_act_probs = tf.nn.softmax(self.old_act_logits, axis=-1)

        self.current_value = self.v_net(
            s_current_feature,
            scope + "_ccurrent")

        self.clip_current_value = self.old_current_value + tf.clip_by_value(
            self.current_value - self.old_current_value, -vf_clip, vf_clip)
        self.current_values = [self.current_value, self.clip_current_value]

    @staticmethod
    def categorical(tensor, num):
        shape = get_shape(tensor)
        if len(shape) == 2:
            return tf.random.categorical(tensor, num)
        elif len(shape) == 3:
            new = tf.reshape(tensor, [-1, shape[-1]])
            sample = tf.random.categorical(new, num)
            return tf.reshape(sample, [shape[0], shape[1], num])
        else:
            raise ValueError(tensor.name + "should have dim 2 or 3")

    def get_return(self):
        return self.ret

    def get_advantage(self):
        return self.advantage

    def get_current_act(self):
        return self.current_act

    def get_current_act_probs(self):
        return self.current_act_probs

    def get_current_act_logits(self):
        return self.current_act_logits

    def get_old_act(self):
        return self.old_act

    def get_old_act_probs(self):
        return self.old_act_probs

    def get_old_act_logits(self):
        return self.old_act_logits

    def get_target_value(self):
        return self.target_value

    def get_current_value(self):
        return self.current_values

    def get_critic_target_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_ctarget"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_target_feature")

    def get_critic_current_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_ccurrent"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_current_feature")

    def get_actor_current_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_acurrent"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_current_feature")

    def get_actor_old_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_aold"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_target_feature")

    def get_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_acurrent"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_ccurrent"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_current_feature")

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

    def feature_net(self, image, lstm, prev_a, scope="feature"):
        shape = get_shape(image)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            image = tf.reshape(image, [-1] + shape[-3:])
            filter = [16, 32, 32]
            kernel = [(3, 3), (3, 3), (3, 3)]
            stride = [(1, 2), (2, 2), (1, 2)]
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
            c_out, h_out = self.c_in, self.h_in

            feature, c_out, h_out = lstm(
                feature, initial_state=[self.c_in, self.h_in])

        return feature, c_out, h_out

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


class Env(object):
    def __init__(self, act_space, act_repeats, frames, game):
        self.act_space = act_space
        self.act_repeats = act_repeats
        self.act_repeat = random.choice(self.act_repeats)
        self.frames = frames

        self.max_pos = -10000

        self.count = 0

        env = gym_super_mario_bros.make(game)
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)

        s_t = self.resize_image(self.env.reset())

        self.s_t = np.tile(s_t, [1, 1, frames])
        self.s_t1 = None
        self.s = [self.s_t]

        self.a_t = act_space
        self.a = []
        self.a_logits = []
        self.a0 = []
        self.r = []
        self.pos = []

        self.v_cur = []

        self.c_in = []
        self.h_in = []
        self.c_in_t = np.zeros(256, dtype=np.float32)
        self.h_in_t = np.zeros(256, dtype=np.float32)

        self.done = False

    def step(self, a, a_logits, c_in, h_in):
        prev_a = self.a_t
        if self.count % self.act_repeat == 0:
            self.a_t = a
        self.count += 1
        gs_t1, gr_t, gdone, ginfo = self.env.step(self.a_t)
        if not gdone:
            s_t1, r_t, done, info = self.env.step(self.a_t)
            r_t += gr_t
            r_t /= 2.
        else:
            s_t1 = gs_t1
            r_t = gr_t
            done = gdone
            info = ginfo
        r_t /= 15.
        s_t1 = self.resize_image(s_t1)
        channels = s_t1.shape[-1]
        self.s_t1 = np.concatenate([s_t1, self.s_t[:, :, :-channels]], axis=-1)

        self.s.append(self.s_t1)
        self.a.append(self.a_t)
        self.a_logits.append(a_logits)
        self.a0.append(prev_a)
        self.r.append(r_t)
        self.max_pos = max(self.max_pos, info["x_pos"])
        self.pos.append(info["x_pos"])
        if (len(self.pos) > 500) and (
                info["x_pos"] - self.pos[-500] < 5) and (
                self.pos[-500] - info["x_pos"] < 5):
            done = True
        self.done = done

        self.c_in.append(self.c_in_t)
        self.h_in.append(self.h_in_t)
        self.c_in_t = c_in
        self.h_in_t = h_in

        self.s_t = self.s_t1

    def update_v(self, v_cur):
        self.v_cur.append(v_cur)

    def reset(self, force=False):
        if self.done or force:
            self.count = 0
            self.act_repeat = random.choice(self.act_repeats)

            s_t = self.resize_image(self.env.reset())

            self.s_t = np.tile(s_t, [1, 1, self.frames])
            self.s_t1 = None
            self.s = [self.s_t]

            self.a_t = self.act_space
            self.a = []
            self.a_logits = []
            self.a0 = []
            self.r = []
            self.pos = []

            self.v_cur = []

            self.c_in = []
            self.h_in = []
            self.c_in_t = np.zeros(256, dtype=np.float32)
            self.h_in_t = np.zeros(256, dtype=np.float32)

            self.done = False

    def get_state(self):
        return self.s_t

    def get_act(self):
        return self.a_t

    def get_max_pos(self):
        return self.max_pos

    def reset_max_pos(self):
        self.max_pos = -10000

    def get_c_in(self):
        return self.c_in_t

    def get_h_in(self):
        return self.h_in_t

    def get_history(self, force=False):
        if self.done or force:
            s = self.s[:-1]
            s1 = self.s[1:]
            a = self.a
            a_logits = self.a_logits
            a0 = self.a0
            r = self.r
            if self.done:
                gaes = get_gaes(None, r, self.v_cur, self.v_cur[1:] + [0], 0.99, 0.95)[0]
                ret = np.array(gaes) + np.array(self.v_cur)
                seg = Seg(s, a, a_logits, a0, r, s1, gaes,
                          ret, self.v_cur, self.v_cur[1:] + [0], self.c_in, self.h_in)
                return seg

            if force and len(r) > 1:
                gaes = get_gaes(None, r[:-1], self.v_cur[:-1], self.v_cur[1:], 0.99, 0.95)[0]
                ret = np.array(gaes) + np.array(self.v_cur[:-1])
                seg = Seg(s[:-1], a[:-1], a_logits[:-1], a0[:-1], r[:-1], s1[:-1], gaes,
                          ret, self.v_cur[:-1], self.v_cur[1:], self.c_in[:-1], self.h_in[:-1])
                return seg

        return None

    @staticmethod
    def resize_image(image, size=84):
        image = Image.fromarray(image)
        image = image.convert("L")
        image = image.resize((size, size))
        image = np.array(image)
        image = image / 255.
        return image[:, :, None]


def padding(input, seqlen, dtype):
    input = np.array(input, dtype=dtype)
    if len(input) >= seqlen:
        return input
    shape = input.shape
    pad = np.tile(
        np.zeros_like(input[0:1], dtype=dtype),
        [seqlen - shape[0]] + (len(shape) - 1) * [1])
    return np.concatenate([input, pad], axis=0)


def worker(**kwargs):
    action_repeats = [1, 2, 3]

    BASE_DIR = kwargs.get("BASE_DIR")
    CKPT_DIR = kwargs.get("CKPT_DIR")
    DATA_DIR = kwargs.get("DATA_DIR")
    max_segs = kwargs.get("max_segs")
    sPREID = kwargs.get("sPREID")
    seqlen = kwargs.get("seqlen")
    frames = kwargs.get("frames")
    MAX_STEPS = kwargs.get("MAX_STEPS")
    CLIP = kwargs.get("CLIP")

    logging.basicConfig(filename=os.path.join(BASE_DIR, "Workerlog"), level="INFO")

    sess = tf.Session()

    phs = dict()

    phs["s"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, frames])
    phs["prev_a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["a_logits"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 7])
    phs["s1"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, frames])
    phs["ret"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["adv"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["v_cur"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["v_tar"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["c_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 256])
    phs["h_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 256])
    phs["slots"] = tf.placeholder(dtype=tf.float32, shape=[None, None])

    lstm = tf.compat.v1.keras.layers.LSTM(
        256, return_sequences=True, return_state=True, name="lstm")

    model = Model(7, CLIP, lstm, "agent", **phs)

    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=6)

    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    envs = []
    games = ["SuperMarioBros-1-1-v0",
             "SuperMarioBros-2-1-v0",
             "SuperMarioBros-4-1-v0",
             "SuperMarioBros-5-1-v0"]

    for i in range(4):
        env = Env(7, action_repeats, frames, games[i])
        envs.append(env)

    i = 0
    POSTID = 0
    print("  %s START !!!" % sPREID)

    while i < MAX_STEPS:
        i += 1
        # start = time.time()
        # if episodic % 1 == 1:
        #     env.render()
        if (i + 1) % 1000 == 0:
            print("steps %d" % (i + 1))

        _s_t_batch = [env.get_state()[None, :, :, :] for env in envs]
        _a_t_batch = [[env.get_act()] for env in envs]
        _c_in_batch = [env.get_c_in() for env in envs]
        _h_in_batch = [env.get_h_in() for env in envs]

        _a_t_new, _a_t_logits, _v_cur, _c_out_batch, _h_out_batch = sess.run(
            [model.get_current_act(),
             model.get_current_act_logits(),
             model.current_value,
             model.c_out,
             model.h_out],
            feed_dict={model.s_t: _s_t_batch,
                       model.previous_actions: _a_t_batch,
                       model.c_in: _c_in_batch,
                       model.h_in: _h_in_batch})

        [env.step(
            _a_t_new[i][0],
            _a_t_logits[i][0],
            _c_out_batch[i],
            _h_out_batch[i]
        ) for (i, env) in enumerate(envs)]

        [env.update_v(_v_cur[i][0]) for (i, env) in enumerate(envs)]

        force = False
        if i == MAX_STEPS - 1:
            force = True

        segs = [env.get_history(force) for env in envs]
        for seg in segs:
            if seg is not None:
                while len(seg[0]) > seqlen // 2:
                    sPOSTID = str(POSTID)
                    sPOSTID = (4 - len(sPOSTID)) * "0" + sPOSTID
                    DATAID = sPREID + "_" + sPOSTID
                    POSTID += 1

                    next_seg = dict()

                    next_seg["s"] = padding(seg.s[:seqlen], seqlen, np.float32)
                    next_seg["a"] = padding(seg.a[:seqlen], seqlen, np.int32)
                    next_seg["prev_a"] = padding(seg.prev_a[:seqlen], seqlen, np.int32)
                    next_seg["a_logits"] = padding(seg.a_logits[:seqlen], seqlen, np.float32)
                    next_seg["r"] = padding(seg.r[:seqlen], seqlen, np.float32)
                    next_seg["s1"] = padding(seg.s1[:seqlen], seqlen, np.float32)
                    next_seg["ret"] = padding(seg.ret[:seqlen], seqlen, np.float32)
                    next_seg["adv"] = padding(seg.gaes[:seqlen], seqlen, np.float32)
                    next_seg["v_cur"] = padding(seg.v_cur[:seqlen], seqlen, np.float32)
                    next_seg["v_tar"] = padding(seg.v_tar[:seqlen], seqlen, np.float32)
                    next_seg["c_in"] = padding(seg.c_in[0], seqlen, np.float32)
                    next_seg["h_in"] = padding(seg.h_in[0], seqlen, np.float32)
                    next_seg["slots"] = padding(
                        seqlen // 2 * [0] + (len(seg.s[:seqlen]) - seqlen // 2) * [1],
                        seqlen, np.int32)

                    while len(glob.glob(os.path.join(DATA_DIR, "*.seg"))
                              ) + len(glob.glob(os.path.join(DATA_DIR, "*.tmp"))
                                      ) > max_segs:
                        time.sleep(1)

                    with pa.OSFile(os.path.join(DATA_DIR, DATAID + ".tmp"), "wb") as f:
                        f.write(pack(next_seg))
                    try:
                        os.rename(os.path.join(DATA_DIR, DATAID + ".tmp"),
                                  os.path.join(DATA_DIR, DATAID + ".seg"))
                    except Exception as e:
                        logging.warning(e)

                    seg = Seg(*[t[seqlen // 2:] for t in seg])

        [env.reset(force) for env in envs]

    max_pos = [env.get_max_pos() for env in envs]
    logging.info(sPREID + " Max Position " + " ".join([str(t) for t in max_pos]))
    print("Max Position", max_pos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-BASE_DIR", type=str)
    parser.add_argument("-CKPT_DIR", type=str)
    parser.add_argument("-DATA_DIR", type=str)
    parser.add_argument("-max_segs", type=int)
    parser.add_argument("-sPREID", type=str)
    parser.add_argument("-seqlen", type=int)
    parser.add_argument("-frames", type=int)
    parser.add_argument("-action_repeat", type=int)
    parser.add_argument("-MAX_STEPS", type=int)
    parser.add_argument("-CLIP", type=float)
    args = parser.parse_args()
    worker(**args.__dict__)
    pass
