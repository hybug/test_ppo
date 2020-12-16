import time
import random
import math
import numpy as np

from collections import namedtuple


def padding(input, seqlen, dtype):
    input = np.array(input, dtype=dtype)
    if len(input) >= seqlen:
        return input
    shape = input.shape
    pad = np.tile(
        np.zeros_like(input[0:1], dtype=dtype),
        [seqlen - shape[0]] + (len(shape) - 1) * [1])
    return np.concatenate([input, pad], axis=0)


Seg = namedtuple("Seg", ["s", "a", "r", "state_in", "mask"])


def _warp_env():
    import random
    from PIL import Image
    import gym

    class Env(object):
        def __init__(self, **kwargs):
            self.act_space = kwargs.get("act_space")
            self.state_size = kwargs.get("state_size")
            self.burn_in = kwargs.get("burn_in")
            self.seqlen = kwargs.get("seqlen")
            self.n_step = kwargs.get("n_step")
            self.frames = kwargs.get("frames")

            self.game = kwargs.get("game")

            self.env = gym.make(self.game)

            self.r = [0]
            self.R = 0
            self.count = 0
            self.done = True
            self.reset()

        def step(self, a, state_in):
            self.count += 1
            if random.random() < self.epsilon:
                a = random.randint(0, self.act_space - 1)
            self.a_t = a
            s_t1, r_t, done, info = self.env.step(self.a_t)
            s_t1 = self.resize_image(s_t1)
            channels = s_t1.shape[-1]
            self.s_t = np.concatenate([s_t1, self.s_t[:, :, :-channels]], axis=-1)

            self.s.append(self.s_t)
            self.a.append(self.a_t)

            self.R += r_t
            self.r.append(r_t)

            self.done = done
            if self.done:
                self.mask.append(0)
            else:
                self.mask.append(1)

            self.state_in.append(state_in)

            """
            get segs
            """
            segs = self.get_history()

            self.reset()

            return segs

        def reset(self):
            if self.done:
                print("  Game %s, Count %d, Return %.2f" % (self.game, self.count, self.R))
                self.epsilon = np.power(0.4, random.uniform(1, 8))
                self.count = 0
                self.R = 0
                s_t = self.env.reset()
                s_t = self.resize_image(s_t)

                self.s_t = np.tile(s_t, [1, 1, self.frames])
                self.s = [np.zeros_like(self.s_t) for i in range(self.burn_in)] + [self.s_t]

                self.a_t = random.randint(0, self.act_space - 1)
                self.a = [random.randint(0, self.act_space - 1) for i in range(self.burn_in)] + [self.a_t]
                self.r = [0] * self.burn_in + [0]
                self.mask = [1] * self.burn_in + [1]

                state_in = np.zeros(self.state_size, dtype=np.float32)
                self.state_in = [state_in] * self.burn_in + [state_in]

                self.done = False

        def get_state(self):
            return self.s_t

        def get_act(self):
            return self.a_t

        def get_reward(self):
            return self.r[-1]

        def get_state_in(self):
            return self.state_in[-1]

        def get_history(self):
            segs = []
            t = self.burn_in
            if self.done:
                seg = Seg(self.s[t:], self.a[t:],
                          self.r[t:],
                          self.state_in[t:], self.mask[t:])
                segs += self.postprocess(seg)
            elif len(self.s) >= self.burn_in + self.seqlen + self.n_step:
                cut = self.burn_in + self.seqlen + self.n_step
                seg = Seg(self.s[:cut], self.a[:cut],
                          self.r[:cut], self.state_in[:cut], self.mask[:cut])

                self.s = self.s[t:]
                self.a = self.a[t:]
                self.r = self.r[t:]
                self.state_in = self.state_in[t:]
                self.mask = self.mask[t:]

                return [self.postprocess_one_seg(seg)]
            return segs

        def postprocess_one_seg(self, seg):
            seqlen = self.seqlen + self.burn_in + self.n_step

            next_seg = dict()

            next_seg["s"] = padding(seg.s[:seqlen], seqlen, np.uint8)
            next_seg["a"] = padding(seg.a[:seqlen], seqlen, np.int32)
            next_seg["r"] = padding(seg.r[:seqlen], seqlen, np.float32)
            next_seg["state_in"] = np.array(seg.state_in[0], np.float32)
            next_seg["mask"] = padding(seg.mask[:seqlen], seqlen, np.int32)

            return next_seg

        def postprocess(self, seg):
            """
            postprocess the seg for training
            :author lhw
            """
            burn_in = self.burn_in
            seg_results = []
            if seg is not None:
                while len(seg[0]) > burn_in + self.n_step:
                    next_seg = self.postprocess_one_seg(seg)
                    seg_results.append(next_seg)
                    seg = Seg(*[t[burn_in:] for t in seg])
            return seg_results

        @staticmethod
        def resize_image(image, size=84):
            image = Image.fromarray(image)
            image = image.convert("L")
            image = image.resize((size, size))
            image = np.array(image, np.uint8)
            return image[:, :, None]

    class Envs(object):
        def __init__(self, **kwargs):
            self.envs = [Env(**kwargs) for i in range(32)]

        def step(self, sess, model):
            fd = self.get_feed_dict(model)

            a, state_in = sess.run(
                [model.current_act,
                 model.state_out],
                feed_dict=fd)

            segs = []
            for i, env in enumerate(self.envs):
                segs += env.step(
                    a[i][0], state_in[i])

            return segs

        def get_feed_dict(self, model):
            fd = dict()
            fd[model.s] = [[env.get_state()] for env in self.envs]
            fd[model.a] = [[env.get_act()] for env in self.envs]
            fd[model.r] = [[env.get_reward()] for env in self.envs]
            fd[model.state_in] = [env.get_state_in() for env in self.envs]
            return fd

    return Envs


def build_env(kwargs):
    Envs = _warp_env()

    envs = Envs(**kwargs)

    return envs
