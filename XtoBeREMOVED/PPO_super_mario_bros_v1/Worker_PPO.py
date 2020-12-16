# coding: utf-8

import sys
sys.path.append("/opt/tiger/test_ppo")

import numpy as np
import os
import logging
from collections import namedtuple
import time
import glob
import random
from PIL import Image
import pyarrow as pa
import zmq

from utils import get_gaes
from utils import pack

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

os.environ["CUDA_VISIBLE_DEVICES"] = ""
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


def padding(input, seqlen, dtype):
    input = np.array(input, dtype=dtype)
    if len(input) >= seqlen:
        return input
    shape = input.shape
    pad = np.tile(
        np.zeros_like(input[0:1], dtype=dtype),
        [seqlen - shape[0]] + (len(shape) - 1) * [1])
    return np.concatenate([input, pad], axis=0)


def Worker_Q(queue_in,
             address,
             parallel,
             BASE_DIR,
             DATA_DIR,
             max_segs,
             server_id,
             worker_id,
             game,
             frames,
             seqlen,
             burn_in,
             act_space,
             state_size,
             **kwargs):
    PREID = 0
    games = game.split("\t")
    seqlen = seqlen + burn_in

    action_repeats = [1, 2]
    envs = [Env(act_space, action_repeats, frames, state_size,
                games[i % len(games)]
                ) for i in range(parallel)]

    logging.basicConfig(filename=os.path.join(BASE_DIR, "Workerlog"), level="INFO")

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    identity = str(worker_id)
    socket.identity = identity.encode("ascii")
    socket.connect(address)
    poll = zmq.Poller()
    poll.register(socket, zmq.POLLIN)

    databack = [[], [], [], []]
    for i in range(parallel):
        databack[0].append([envs[i].get_state()])
        databack[1].append([envs[i].get_act()])
        databack[2].append([envs[i].r[-1]])
        databack[3].append(envs[i].get_state_in())
    socket.send(pack((worker_id, databack)), copy=False)

    while True:
        a_b, a_logits_b, state_in_b, v_cur_b = queue_in.get()
        databack = [[], [], [], []]
        for i in range(parallel):
            a = a_b[i][0]
            a_logits = a_logits_b[i][0]
            state_in = state_in_b[i]
            v_cur = v_cur_b[i][0]
            envs[i].step(a, a_logits, state_in)
            envs[i].update_v(v_cur)
            seg = envs[i].get_history()
            if seg is not None:
                while len(seg[0]) > burn_in:
                    sPOSTID = (4 - len(str(worker_id))) * "0" + str(worker_id) + "_" + (
                            4 - len(str(i))) * "0" + str(i) + "_" + (
                                      4 - len(str(server_id))) * "0" + str(server_id)
                    sPREID = str(PREID)
                    sPREID = (12 - len(sPREID)) * "0" + sPREID
                    DATAID = sPREID + "_" + sPOSTID
                    PREID += 1

                    next_seg = dict()

                    next_seg["s"] = padding(seg.s[:seqlen], seqlen, np.float32)
                    next_seg["a"] = padding(seg.a[1:seqlen + 1], seqlen, np.int32)
                    next_seg["prev_a"] = padding(seg.a[:seqlen], seqlen, np.int32)
                    next_seg["a_logits"] = padding(seg.a_logits[:seqlen], seqlen, np.float32)
                    next_seg["prev_r"] = padding(seg.r[:seqlen], seqlen, np.float32)
                    next_seg["r"] = padding(seg.r[1:seqlen+1], seqlen, np.float32)
                    next_seg["adv"] = padding(seg.gaes[:seqlen], seqlen, np.float32)
                    next_seg["v_cur"] = padding(seg.v_cur[:seqlen], seqlen, np.float32)
                    next_seg["state_in"] = np.array(seg.state_in[0], np.float32)
                    next_seg["slots"] = padding(len(seg.s[:seqlen]) * [1], seqlen, np.int32)

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

                    seg = Seg(*[t[burn_in:] for t in seg])
            envs[i].reset()

            databack[0].append([envs[i].get_state()])
            databack[1].append([envs[i].get_act()])
            databack[2].append([envs[i].r[-1]])
            databack[3].append(envs[i].get_state_in())
        socket.send(pack((worker_id, databack)), copy=False)
