# coding: utf-8

import sys

sys.path.append(".../")

import tensorflow as tf
import numpy as np
import os
import logging
from collections import namedtuple
import random
from PIL import Image

from utils import get_gaes, get_shape
from infer import categorical
from module import RMCRNN, TmpHierRMCRNN

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

logging.getLogger('tensorflow').setLevel(logging.ERROR)
Seg = namedtuple("Seg", ["s", "a", "a_logits", "r", "gaes", "v_cur", "state_in"])


class Env(object):
    def __init__(self, act_space, act_repeats, frames, state_size, game):
        self.act_space = act_space
        self.act_repeats = act_repeats
        self.act_repeat = random.choice(self.act_repeats)
        self.frames = frames
        self.state_size = state_size
        self.game = game

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
        self.r = [0]
        self.pos = []

        self.v_cur = []

        state_in = np.zeros(self.state_size, dtype=np.float32)
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
            max_pos = self.max_pos
            self.max_pos = -10000
            logging.info("  Max Position  %s : %d" % (
                self.game, max_pos))
            self.count = 0
            self.act_repeat = random.choice(self.act_repeats)

            s_t = self.resize_image(self.env.reset())

            self.s_t = np.tile(s_t, [1, 1, self.frames])
            self.s = [self.s_t]

            self.a_t = random.randint(0, self.act_space - 1)
            self.a = [self.a_t]
            self.a_logits = []
            self.r = [0]
            self.pos = []

            self.v_cur = []

            state_in = np.zeros(self.state_size, dtype=np.float32)
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
                gaes = get_gaes(None, self.r, self.v_cur, self.v_cur[1:] + [0], 0.99, 0.95)[0]
                seg = Seg(self.s, self.a, self.a_logits, self.r, gaes, self.v_cur, self.state_in)
                return seg
            if force and len(self.r) > 1:
                gaes = get_gaes(None, self.r[:-1], self.v_cur[:-1], self.v_cur[1:], 0.99, 0.95)[0]
                seg = Seg(self.s[:-1], self.a[:-1], self.a_logits[:-1], self.r[:-1], gaes,
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


class Model(object):
    def __init__(self,
                 act_space,
                 rnn,
                 use_rmc,
                 use_hrmc,
                 use_reward_prediction,
                 after_rnn,
                 use_pixel_control,
                 use_pixel_reconstruction,
                 scope="agent",
                 **kwargs):
        self.act_space = act_space
        self.scope = scope
        self.use_rmc = use_rmc
        self.use_hrmc = use_hrmc

        self.s_t = kwargs.get("s")
        self.previous_actions = kwargs.get("prev_a")
        self.prev_r = kwargs.get("prev_r")
        self.state_in = kwargs.get("state_in")

        prev_a = tf.one_hot(
            self.previous_actions, depth=act_space, dtype=tf.float32)

        self.feature, self.cnn_feature, self.image_feature, self.state_out = self.feature_net(
            self.s_t, rnn, prev_a, self.prev_r, self.state_in, scope + "_current_feature")

        if self.use_hrmc:
            self.p_zs = self.feature["p_zs"]
            self.p_mus = self.feature["p_mus"]
            self.p_sigmas = self.feature["p_sigmas"]
            self.q_mus = self.feature["q_mus"]
            self.q_sigmas = self.feature["q_sigmas"]
            self.feature = self.feature["q_zs"]

        self.current_act_logits = self.a_net(
            self.feature, scope + "_acurrent")
        self.current_act = tf.squeeze(
            categorical(self.current_act_logits), axis=-1)

        self.current_value = self.v_net(
            self.feature,
            scope + "_ccurrent")

        advantage = kwargs.get("adv", None)
        if advantage is not None:
            self.old_current_value = kwargs.get("v_cur")
            self.ret = advantage + self.old_current_value

            self.a_t = kwargs.get("a")
            self.old_act_logits = kwargs.get("a_logits")
            self.r_t = kwargs.get("r")

            self.adv_mean = tf.reduce_mean(advantage, axis=[0, 1])
            advantage -= self.adv_mean
            self.adv_std = tf.math.sqrt(tf.reduce_mean(advantage ** 2, axis=[0, 1]))
            self.advantage = advantage / tf.maximum(self.adv_std, 1e-12)

            self.slots = tf.cast(kwargs.get("slots"), tf.float32)

            if use_reward_prediction:
                if after_rnn:
                    self.reward_prediction = self.r_net(self.feature, "r_net")
                else:
                    self.reward_prediction = self.r_net(self.cnn_feature, "r_net")

            if use_pixel_reconstruction:
                self.pixel_reconstruction = self.reconstruct_net(self.feature)

            if use_pixel_control:
                self.pixel_control = self.control_net(self.feature)

    def get_current_act(self):
        return self.current_act

    def get_current_act_logits(self):
        return self.current_act_logits

    def v_net(self, feature, scope):
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

    def a_net(self, feature, scope):
        net = feature
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(
                net,
                get_shape(feature)[-1],
                activation=tf.nn.relu,
                name="dense")
            act_logits = tf.layers.dense(
                net,
                self.act_space,
                activation=None,
                name="a_logits")

        return act_logits

    def r_net(self, feature, scope):
        net = feature
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(
                net,
                get_shape(feature)[-1],
                activation=tf.nn.relu,
                name="dense")
            r_pred = tf.squeeze(
                tf.layers.dense(
                    net,
                    1,
                    activation=None,
                    name="r_pred"),
                axis=-1)

        return r_pred

    def feature_net(self, image, rnn, prev_a, prev_r, state_in, scope="feature"):
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
            image_feature = tf.reshape(
                image, [shape[0], shape[1], new_shape[1], new_shape[2], new_shape[3]])

            feature = tf.reshape(
                image, [shape[0], shape[1], new_shape[1] * new_shape[2] * new_shape[3]])

            cnn_feature = tf.layers.dense(feature, 256, tf.nn.relu, name="feature")
            feature = tf.concat([cnn_feature, prev_a, prev_r[:, :, None]], axis=-1)

            if self.use_hrmc:
                initial_state = tf.split(state_in, [1, -1], axis=-1)
                feature, count_out, state_out = rnn(
                    feature, initial_state=initial_state)
                state_out = tf.concat([count_out, state_out], axis=-1)
            elif self.use_rmc:
                initial_state = [state_in]
                feature, state_out = rnn(
                    feature, initial_state=initial_state)
            else:
                initial_state = tf.split(state_in, 2, axis=-1)
                feature, c_out, h_out = rnn(
                    feature, initial_state=initial_state)
                state_out = tf.concat([c_out, h_out], axis=-1)

        return feature, cnn_feature, image_feature, state_out

    def reconstruct_net(self, feature, scope="reconstruct"):
        shape = get_shape(feature)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            feature = tf.reshape(feature, [-1, shape[-1]])
            feature = tf.layers.dense(feature, 800, tf.nn.relu, name="feature")
            image = tf.reshape(feature, [-1, 5, 5, 32])
            filter = [16, 32, 32]
            size = [(84, 82), (40, 38), (18, 7)]
            kernel = [(3, 3), (3, 3), (5, 3)]
            stride = [(1, 2), (1, 2), (2, 1)]
            for i in range(len(filter) - 1, -1, -1):
                image = self.resblock(
                    image, "res0_%d" % i)

                image = tf.image.resize_nearest_neighbor(
                    image, [size[i][1], size[i][1]])

                output_channels = filter[i - 1] if i > 0 else 1
                input_channels = filter[i]
                image = tf.nn.conv2d_transpose(
                    image,
                    filter=tf.get_variable(
                        name="deconv_%d" % i,
                        shape=[kernel[i][0], kernel[i][0], output_channels, input_channels]),
                    output_shape=[get_shape(feature)[0], size[i][0], size[i][0], output_channels],
                    strides=stride[i][0],
                    padding="VALID")

            image = tf.reshape(image, shape=[shape[0], shape[1]] + get_shape(image)[-3:])

        return image

    def control_net(self, feature, scope="pixel_control"):
        shape = get_shape(feature)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            feature = tf.reshape(feature, [-1, shape[-1]])
            feature = tf.layers.dense(feature, 7 * 7 * 32, tf.nn.relu, name="feature")
            image = tf.reshape(feature, [-1, 7, 7, 32])
            image = tf.nn.conv2d_transpose(
                image,
                filter=tf.get_variable(
                    name="deconv",
                    shape=[9, 9, 32, 32]),
                output_shape=[get_shape(feature)[0], 21, 21, 32],
                strides=2,
                padding="VALID")
            image = tf.nn.relu(image)
            image = tf.nn.conv2d_transpose(
                image,
                filter=tf.get_variable(
                    name="control",
                    shape=[4, 4, self.act_space, 32]),
                output_shape=[get_shape(feature)[0], 21, 21, self.act_space],
                strides=1,
                padding="SAME")

            image = tf.reshape(image, shape=[shape[0], shape[1]] + get_shape(image)[-3:])

        return image

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


def run():
    CKPT_DIR = "/".join(os.getcwd().split("/")[:-2]) + "/ckpt/ppo23"

    frames = 1
    action_repeats = [1, 2]
    MAX_STEPS = 320000
    act_space = 12
    use_rmc = False
    use_hrmc = True
    use_reward_prediction = False
    use_pixel_control = False
    use_pixel_reconstruction = False
    after_rnn = False

    sess = tf.Session()

    phs = dict()

    phs["s"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, frames])
    phs["prev_a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["prev_r"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    # phs["a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    # phs["a_logits"] = tf.placeholder(dtype=tf.float32, shape=[None, None, act_space])
    # phs["adv"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["v_cur"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    # phs["slots"] = tf.placeholder(dtype=tf.float32, shape=[None, None])

    if use_hrmc:
        state_size = 1 + 2 * (4 + 4) * 4 * 64
        phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        lstm = TmpHierRMCRNN(
            4, 64, 4, 4, return_sequences=True, return_state=True, name="hrmcrnn")
    elif use_rmc:
        state_size = 64 * 4 * 4
        phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 64 * 4 * 4])
        lstm = RMCRNN(
            64, 4, 4, return_sequences=True, return_state=True, name="rmcrnn")
    else:
        state_size = 256 * 2
        phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 256 * 2])
        lstm = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")

    model = Model(
        act_space, lstm, use_rmc, use_hrmc,
        use_reward_prediction, after_rnn, use_pixel_control,
        use_pixel_reconstruction, "agent", **phs)

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
        env = Env(12, action_repeats, frames, state_size, games[i])
        envs.append(env)

    while True:
        for i in range(MAX_STEPS):
            _s_t_batch = [env.get_state()[None, :, :, :] for env in envs]
            _a_t_batch = [[env.get_act()] for env in envs]
            _r_t_batch = [[env.r[-1]] for env in envs]
            _state_in_batch = [env.get_state_in() for env in envs]

            _a_t_new, _a_t_logits, _v_cur, _state_out_batch = sess.run(
                [model.get_current_act(),
                 model.get_current_act_logits(),
                 model.current_value,
                 model.state_out],
                feed_dict={model.s_t: _s_t_batch,
                           model.previous_actions: _a_t_batch,
                           model.prev_r: _r_t_batch,
                           model.state_in: _state_in_batch})

            # _a_t_new = np.argmax(_a_t_logits, axis=-1)

            [env.step(
                _a_t_new[i][0],
                _a_t_logits[i][0],
                _state_out_batch[i]
            ) for (i, env) in enumerate(envs)]

            [env.update_v(_v_cur[i][0]) for (i, env) in enumerate(envs)]

            force = False
            if i == MAX_STEPS - 1:
                force = True

            [env.reset(force) for env in envs]


if __name__ == '__main__':
    run()
    pass
