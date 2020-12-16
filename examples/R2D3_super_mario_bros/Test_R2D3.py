# coding: utf-8

import sys
import pickle
import time
sys.path.append(".../")

import tensorflow as tf
import numpy as np
import os
import logging
import random
from PIL import Image

from examples.R2D3_super_mario_bros.policy_graph import build_evaluator_model

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# logging.getLogger('tensorflow').setLevel(logging.ERROR)


class Env(object):
    def __init__(self, game, **kwargs):
        self.act_space = kwargs.get("act_space")
        self.state_size = kwargs.get("state_size")
        self.burn_in = kwargs.get("burn_in")
        self.seqlen = kwargs.get("seqlen")
        self.n_step = kwargs.get("n_step")
        self.use_soft = kwargs.get("use_soft")
        self.frames = kwargs.get("frames")
        self.sample_epsilon_per_step = kwargs.get("sample_epsilon_per_step")

        self.epsilon = np.power(0.4, random.uniform(4, 8))
        self.game = game

        self.count = 0
        self.count_maxpos = []

        env = gym_super_mario_bros.make(game)
        if self.act_space == 7:
            self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        elif self.act_space == 12:
            self.env = JoypadSpace(env, COMPLEX_MOVEMENT)

        self.max_pos = -10000
        self.done = True
        self.reset()

    def step(self, a, state_in):
        maxpos = self.reset()

        self.count += 1
        if not self.use_soft:
            if self.sample_epsilon_per_step:
                self.epsilon = np.power(0.4, random.uniform(4, 8))
            if random.random() < self.epsilon:
                a = random.randint(0, self.act_space - 1)
        self.a_t = a
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
        s_t1 = self.resize_image(s_t1)
        channels = s_t1.shape[-1]
        self.s_t = np.concatenate([s_t1, self.s_t[:, :, :-channels]], axis=-1)

        self.s.append(self.s_t)
        self.a.append(self.a_t)
        self.r.append(r_t)
        self.max_pos = max(self.max_pos, info["x_pos"])
        self.pos.append(info["x_pos"])
        if (len(self.pos) > 100) and (
                info["x_pos"] - self.pos[-100] < 5) and (
                self.pos[-100] - info["x_pos"] < 5):
            done = True
        self.done = done
        if self.done:
            self.mask.append(0)
        else:
            self.mask.append(1)

        self.state_in.append(state_in)

        """
        get segs
        """
        # segs = self.get_history()
        #
        # return segs
        return maxpos

    def reset(self):
        if self.done:
            self.count_maxpos.append(self.max_pos)
            print(self.game, self.max_pos, len(self.count_maxpos[1:]), np.mean(self.count_maxpos[1:]))
            self.epsilon = np.power(0.4, random.uniform(4, 8))

            self.count = 0

            s_t = self.resize_image(self.env.reset())

            self.s_t = np.tile(s_t, [1, 1, self.frames])
            self.s = [self.s_t]

            self.a_t = random.randint(0, self.act_space - 1)
            self.a = [self.a_t]
            self.r = [0]
            self.mask = [1]

            self.max_pos = -10000
            self.pos = []

            state_in = np.zeros(self.state_size, dtype=np.float32)
            self.state_in = [state_in]

            self.done = False
            return self.count_maxpos
        return None

    def get_state(self):
        return self.s_t

    def get_act(self):
        return self.a_t

    def get_reward(self):
        return self.r[-1]

    def get_max_pos(self):
        return self.max_pos

    def get_state_in(self):
        return self.state_in[-1]

    @staticmethod
    def resize_image(image, size=84):
        image = Image.fromarray(image)
        image = image.convert("L")
        image = image.resize((size, size))
        image = np.array(image, np.uint8)
        return image[:, :, None]


def run():
    CKPT_DIR = "/".join(os.getcwd().split("/")[:-2]) + "/ckpt/r2d3_0"

    kwargs = dict()
    kwargs["use_soft"] = 0
    kwargs["frames"] = 1
    kwargs["burn_in"] = 32
    kwargs["seqlen"] = 32
    kwargs["act_space"] = 12
    kwargs["n_step"] = 5
    kwargs["use_hrnn"] = 1
    kwargs["sample_epsilon_per_step"] = 0
    kwargs["gamma"] = 0.99
    kwargs["use_reward_prediction"] = 1
    kwargs["after_rnn"] = 1
    kwargs["use_pixel_control"] = 1
    kwargs["image_size"] = 84
    kwargs["time_scale"] = 4
    if kwargs["use_hrnn"]:
        kwargs["state_size"] = 1 + (8 + 2 + 8) * 4 * 64
    else:
        kwargs["state_size"] = 256 * 2

    model = build_evaluator_model(kwargs)

    sess = tf.Session()

    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=6)

    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    saver.restore(sess, os.path.join(CKPT_DIR, ckpt.model_checkpoint_path.split("/")[-1]))

    envs = []
    games = ["SuperMarioBros-%d-%d-v0" % (i, j) for i in range(8, 9) for j in range(1, 5)]
    for i in range(len(games)):
        env = Env(games[i], **kwargs)
        envs.append(env)

    start = time.time()
    while True:
        if time.time() - start > 3600:
            break
        fd = dict()
        fd[model.s] = [[env.get_state()] for env in envs]
        fd[model.a] = [[env.get_act()] for env in envs]
        fd[model.r] = [[env.get_reward()] for env in envs]
        fd[model.state_in] = [env.get_state_in() for env in envs]

        a, state_in = sess.run(
            [model.current_act, model.state_out],
            feed_dict=fd)

        for i, env in enumerate(envs):
            t = env.step(a[i][0], state_in[i])
            if t is not None:
                with open(os.path.join(CKPT_DIR, games[i] + ".pkl"), "wb") as f:
                    pickle.dump(t, f)


if __name__ == '__main__':
    run()
    # draw()
    pass
