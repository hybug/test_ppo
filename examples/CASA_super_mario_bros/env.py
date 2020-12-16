import time
import random
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


Seg = namedtuple("Seg", ["s", "a", "a_logits", "r", "v_cur", "state_in", "mask"])


def _warp_env():
    import random
    from PIL import Image
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

    class Env(object):
        def __init__(self, game, **kwargs):
            self.act_space = kwargs.get("act_space")
            self.state_size = kwargs.get("state_size")
            self.burn_in = kwargs.get("burn_in")
            self.seqlen = kwargs.get("seqlen")
            self.n_step = kwargs.get("n_step")
            self.frames = kwargs.get("frames")
            self.replay = kwargs.get("replay")
            self.use_epsilon_greedy = kwargs.get("use_epsilon_greedy")

            self.game = game

            self.count = 0

            env = gym_super_mario_bros.make(game)
            if self.act_space == 7:
                self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
            elif self.act_space == 12:
                self.env = JoypadSpace(env, COMPLEX_MOVEMENT)

            self.max_pos = -10000
            self.done = True
            self.reset()

        def step(self, a, a_logits, v_cur, state_in):
            self.count += 1
            if self.use_epsilon_greedy:
                a = np.argmax(a_logits)
                a_logits = self.epsilon / self.act_space * np.ones(self.act_space)
                a_logits[a] += (1 - self.epsilon)
                a_logits = np.log(a_logits)
                if random.random() < self.epsilon:
                    a = random.randint(0, self.act_space - 1)
            self.a_t = a
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
            r_t /= 15.0
            s_t1 = self.resize_image(s_t1)
            channels = s_t1.shape[-1]
            self.s_t = np.concatenate([s_t1, self.s_t[:, :, :-channels]], axis=-1)

            self.s.append(self.s_t)
            self.a.append(self.a_t)
            self.a_logits.append(a_logits)
            self.r.append(r_t)
            self.v_cur.append(v_cur)
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
            segs = self.get_history()

            self.reset()

            return segs

        def reset(self):
            if self.done:
                print(self.game, self.max_pos)

                self.count = 0
                self.epsilon = 0.4 ** random.uniform(1, 8)

                s_t = self.resize_image(self.env.reset())

                self.s_t = np.tile(s_t, [1, 1, self.frames])
                self.s = [np.zeros_like(self.s_t) for i in range(self.burn_in)] + [self.s_t]

                self.a_t = random.randint(0, self.act_space - 1)
                self.a = [random.randint(0, self.act_space - 1) for i in range(self.burn_in)] + [self.a_t]
                self.a_logits = [np.zeros(self.act_space) for i in range(self.burn_in)]
                self.r = [0] * self.burn_in + [0]
                self.v_cur = [0] * self.burn_in
                self.mask = [1] * self.burn_in + [1]

                self.max_pos = -10000
                self.pos = []

                state_in = np.zeros(self.state_size, dtype=np.float32)
                self.state_in = [state_in] * self.burn_in + [state_in]

                self.done = False

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

        def get_history(self):
            segs = []
            t = self.burn_in // self.replay
            if self.done:
                for i in range(self.replay):
                    seg = Seg(self.s[i * t:], self.a[i * t:], self.a_logits[i * t:],
                              self.r[i * t:], self.v_cur[i * t:],
                              self.state_in[i * t:], self.mask[i * t:])
                    segs += self.postprocess(seg)
            elif len(self.s) >= self.burn_in + self.seqlen + self.n_step:
                cut = self.burn_in + self.seqlen + self.n_step
                seg = Seg(self.s[:cut], self.a[:cut], self.a_logits[:cut],
                          self.r[:cut], self.v_cur[:cut], self.state_in[:cut], self.mask[:cut])

                self.s = self.s[t:]
                self.a = self.a[t:]
                self.a_logits = self.a_logits[t:]
                self.r = self.r[t:]
                self.v_cur = self.v_cur[t:]
                self.state_in = self.state_in[t:]
                self.mask = self.mask[t:]

                return [self.postprocess_one_seg(seg)]
            return segs

        def postprocess_one_seg(self, seg):
            seqlen = self.seqlen + self.burn_in + self.n_step

            next_seg = dict()

            next_seg["s"] = padding(seg.s[:seqlen], seqlen, np.uint8)
            next_seg["a"] = padding(seg.a[:seqlen], seqlen, np.int32)
            next_seg["a_logits"] = padding(seg.a_logits[:seqlen], seqlen, np.float32)
            next_seg["r"] = padding(seg.r[:seqlen], seqlen, np.float32)
            next_seg["v_cur"] = padding(seg.v_cur[:seqlen], seqlen, np.float32)
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
        def __init__(self, games, **kwargs):
            self.envs = [Env(game, **kwargs) for game in games]

        def step(self, sess, model):
            fd = self.get_feed_dict(model)

            a, a_logits, v_cur, state_in = sess.run(
                [model.current_act,
                 model.current_act_logits,
                 model.current_value,
                 model.state_out],
                feed_dict=fd)

            segs = []
            for i, env in enumerate(self.envs):
                segs += env.step(
                    a[i][0], a_logits[i][0], v_cur[i][0], state_in[i])

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
    games = ["SuperMarioBros-%d-%d-v0" % (i, j) for i in range(1, 9) for j in range(1, 5)]
    random.shuffle(games)

    envs = Envs(games, **kwargs)

    return envs
