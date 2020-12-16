# coding: utf-8

import tensorflow as tf
import numpy as np
import os
import logging
from collections import namedtuple
import random
from PIL import Image

from XtoBeREMOVED.IMPALA_super_mario_bros_v1.Trainer_PPOvtrace import Model

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

logging.getLogger('tensorflow').setLevel(logging.ERROR)
Seg = namedtuple("Seg", ["s", "a", "a_logits", "r", "v_cur", "state_in"])


class Env(object):
    def __init__(self, act_space, act_repeats, frames, game):
        self.act_space = act_space
        self.act_repeats = act_repeats
        self.act_repeat = random.choice(self.act_repeats)
        self.frames = frames

        self.max_pos = -10000

        self.count = 0

        env = gym_super_mario_bros.make(game)
        if self.act_space == 7:
            self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        elif self.act_space == 12:
            self.env = JoypadSpace(env, COMPLEX_MOVEMENT)

        s_t = self.resize_image(self.env.reset())

        self.s_t = np.tile(s_t, [1, 1, frames])
        self.s = [self.s_t]

        self.a_t = random.randint(0, act_space - 1)
        self.a = [self.a_t]
        self.a_logits = []
        self.r = []
        self.pos = []

        self.v_cur = []

        c_in = np.zeros(256, dtype=np.float32)
        h_in = np.zeros(256, dtype=np.float32)
        state_in = np.concatenate([c_in, h_in], axis=-1)
        self.state_in = [state_in]

        self.done = False

    def step(self, a, a_logits, state_in):
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
        self.s_t = np.concatenate([s_t1, self.s_t[:, :, :-channels]], axis=-1)

        self.s.append(self.s_t)
        self.a.append(self.a_t)
        self.a_logits.append(a_logits)
        self.r.append(r_t)
        self.max_pos = max(self.max_pos, info["x_pos"])
        self.pos.append(info["x_pos"])
        if (len(self.pos) > 500) and (
                info["x_pos"] - self.pos[-500] < 5) and (
                self.pos[-500] - info["x_pos"] < 5):
            done = True
        self.done = done

        self.state_in.append(state_in)

    def update_v(self, v_cur):
        self.v_cur.append(v_cur)

    def reset(self, force=False):
        if self.done or force:
            self.count = 0
            self.act_repeat = random.choice(self.act_repeats)

            s_t = self.resize_image(self.env.reset())

            self.s_t = np.tile(s_t, [1, 1, self.frames])
            self.s = [self.s_t]

            self.a_t = random.randint(0, self.act_space - 1)
            self.a = [self.a_t]
            self.a_logits = []
            self.r = []
            self.pos = []

            self.v_cur = []

            c_in = np.zeros(256, dtype=np.float32)
            h_in = np.zeros(256, dtype=np.float32)
            state_in = np.concatenate([c_in, h_in], axis=-1)
            self.state_in = [state_in]

            self.done = False

    def get_state(self):
        return self.s_t

    def get_act(self):
        return self.a_t

    def get_max_pos(self):
        return self.max_pos

    def reset_max_pos(self):
        self.max_pos = -10000

    def get_state_in(self):
        return self.state_in[-1]

    def get_history(self, force=False):
        if self.done or force:
            if self.done:
                seg = Seg(self.s, self.a, self.a_logits, self.r, self.v_cur, self.state_in)
                return seg
            if force and len(self.r) > 1:
                seg = Seg(self.s[:-1], self.a[:-1], self.a_logits[:-1], self.r[:-1],
                          self.v_cur[:-1], self.state_in[:-1])
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
    CKPT_DIR = "../ckpt/vtrace1"

    frames = 1
    action_repeats = [1]
    MAX_STEPS = 320000
    gamma = 0.99
    act_space = 12

    sess = tf.Session()

    phs = dict()

    phs["s"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, frames])
    phs["prev_a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 256 * 2])
    phs["slots"] = tf.placeholder(dtype=tf.float32, shape=[None, None])

    lstm = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")
    model = Model(act_space, lstm, gamma, "agent", **phs)

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
    games = ["SuperMarioBros-%d-%d-v0" % (i, j) for i in [8] for j in [1, 2, 3, 4]]
    for i in range(len(games)):
        env = Env(act_space, action_repeats, frames, games[i])
        envs.append(env)

    while True:
        for i in range(MAX_STEPS):
            _s_t_batch = [env.get_state()[None, :, :, :] for env in envs]
            _a_t_batch = [[env.get_act()] for env in envs]
            _state_in_batch = [env.get_state_in() for env in envs]

            fd = dict()
            fd[phs["s"]] = _s_t_batch
            fd[phs["prev_a"]] = _a_t_batch
            fd[phs["state_in"]] = _state_in_batch
            fd[phs["slots"]] = np.ones_like(_a_t_batch)

            _a_t_new, _a_logits_batch, _state_out_batch = sess.run(
                [model.get_current_act(),
                 model.get_current_act_logits(),
                 model.state_out],
                feed_dict=fd)

            # if np.random.random() > 0.1:
            #     _a_t_new = np.argmax(_a_logits_batch, axis=-1)

            [env.step(
                _a_t_new[i][0],
                _a_logits_batch[i][0],
                _state_out_batch[i]
            ) for (i, env) in enumerate(envs)]

            [env.reset() for env in envs]


if __name__ == '__main__':
    run()
    pass
