# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import logging
from collections import namedtuple
import random
from PIL import Image

from zSAC.Trainer_SAC import Model

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

logging.getLogger('tensorflow').setLevel(logging.ERROR)


Seg = namedtuple("Seg", ["s", "a", "a_logits", "prev_a", "r", "s1", "gaes", "ret", "v_cur", "v_tar", "c_in", "h_in"])


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

        self.a_t = random.randint(0, act_space - 1)
        self.a = []
        self.a_logits = []
        self.a0 = []
        self.r = []
        self.pos = []

        self.v_cur = []

        self.c_in = []
        self.h_in = []
        self.c_in_t = np.zeros(128 * 4, dtype=np.float32)
        self.h_in_t = np.zeros(128 * 4, dtype=np.float32)

        self.done = False

    def step(self, a, c_in, h_in):
        prev_a = self.a_t
        self.count += 1
        if self.count % self.act_repeat == 0:
            self.a_t = a
            self.count = 0
            self.act_repeat = random.choice(self.act_repeats)
        gs_t1, gr_t, gdone, ginfo = self.env.step(self.a_t)
        self.env.render()
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

            self.a_t = random.randint(0, self.act_space - 1)
            self.a = []
            self.a_logits = []
            self.a0 = []
            self.r = []
            self.pos = []

            self.v_cur = []

            self.c_in = []
            self.h_in = []
            self.c_in_t = np.zeros(128 * 4, dtype=np.float32)
            self.h_in_t = np.zeros(128 * 4, dtype=np.float32)

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
            a0 = self.a0
            r = self.r
            seg = Seg(s, a, a0, r, s1, self.c_in, self.h_in)
            return seg
        return None

    @staticmethod
    def resize_image(image, size=84):
        image = Image.fromarray(image)
        image = image.convert("L")
        image = image.resize((size, size))
        image = np.array(image)
        image = image / 255.
        image = np.array(image, np.float32)
        return image[:, :, None]


def run():
    CKPT_DIR = "ckpt/sac0"

    frames = 1
    action_repeats = [1]
    MAX_STEPS = 320000
    CLIP = 1.0

    sess = tf.Session()

    phs = dict()

    phs["s"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, frames])
    phs["prev_a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["r"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["bootstrap_s"] = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, frames])
    phs["c_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 128 * 4])
    phs["h_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 128 * 4])
    phs["slots"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["bootstrap_slots"] = tf.placeholder(dtype=tf.float32, shape=[None])

    with tf.variable_scope("p_lstm", reuse=tf.AUTO_REUSE):
        plstm = tf.compat.v1.keras.layers.LSTM(
            128, return_sequences=True, return_state=True, name="lstm")
    with tf.variable_scope("p_lstm", reuse=tf.AUTO_REUSE):
        qlstm = tf.compat.v1.keras.layers.LSTM(
            128, return_sequences=True, return_state=True, name="lstm")
    with tf.variable_scope("v_lstm", reuse=tf.AUTO_REUSE):
        vlstm = tf.compat.v1.keras.layers.LSTM(
            128, return_sequences=True, return_state=True, name="lstm")
    with tf.variable_scope("v_tar_lstm", reuse=tf.AUTO_REUSE):
        vtarlstm = tf.compat.v1.keras.layers.LSTM(
            128, return_sequences=True, return_state=True, name="lstm")
    model = Model(7, plstm, qlstm, vlstm, vtarlstm, "agent", **phs)

    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=6)

    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    saver.restore(sess, os.path.join(CKPT_DIR, ckpt.model_checkpoint_path.split("/")[-1]))

    envs = []
    games = ["SuperMarioBros-1-1-v0",
             "SuperMarioBros-2-1-v0",
             "SuperMarioBros-4-1-v0",
             "SuperMarioBros-5-1-v0"]
    # games = ["SuperMarioBros-2-3-v0",
    #          "SuperMarioBros-5-2-v0",
    #          "SuperMarioBros-7-1-v0",
    #          "SuperMarioBros-7-3-v0",
    #          "SuperMarioBros-8-1-v0",
    #          "SuperMarioBros-8-2-v0",
    #          "SuperMarioBros-8-3-v0"]
    for i in range(len(games)):
        env = Env(7, action_repeats, frames, games[i])
        envs.append(env)

    while True:
        for i in range(MAX_STEPS):
            _s_t_batch = [env.get_state()[None, :, :, :] for env in envs]
            _a_t_batch = [[env.get_act()] for env in envs]
            _c_in_batch = [env.get_c_in() for env in envs]
            _h_in_batch = [env.get_h_in() for env in envs]

            _a_t_new, _a_t_logits, _c_out_batch, _h_out_batch = sess.run(
                [model.get_current_act(),
                 model.get_current_act_logits(),
                 model.c_out,
                 model.h_out],
                feed_dict={model.s_t: _s_t_batch,
                           model.previous_actions: _a_t_batch,
                           model.c_in: _c_in_batch,
                           model.h_in: _h_in_batch})
            if random.random() > 0.5:
                _a_t_new = np.argmax(_a_t_logits, axis=-1)

            [env.step(
                _a_t_new[i][0],
                _c_out_batch[i],
                _h_out_batch[i]
            ) for (i, env) in enumerate(envs)]

            force = False
            if i == MAX_STEPS - 1:
                force = True

            [env.reset(force) for env in envs]


if __name__ == '__main__':
    run()
    pass
