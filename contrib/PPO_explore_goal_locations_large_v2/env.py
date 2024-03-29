import numpy as np

from collections import namedtuple


# todo, to common
def padding(input, seqlen, dtype):
    input = np.array(input, dtype=dtype)
    if len(input) >= seqlen:
        return input
    shape = input.shape
    pad = np.tile(
        np.zeros_like(input[0:1], dtype=dtype),
        [seqlen - shape[0]] + (len(shape) - 1) * [1])
    return np.concatenate([input, pad], axis=0)


Seg = namedtuple("Seg", ["s", "a", "a_logits", "r", "gaes", "v_cur", "state_in"])


def _warp_env():
    import random
    from PIL import Image
    import gym
    import gym_deepmindlab
    from utils import get_gaes

    class Env(object):
        def __init__(self, act_space, act_repeats, frames, state_size, burn_in, seqlen, game):
            self.act_space = act_space
            self.act_repeats = act_repeats
            self.act_repeat = random.choice(self.act_repeats)
            self.frames = frames
            self.state_size = state_size
            self.game = game
            self.burn_in = burn_in
            self.seqlen = seqlen

            self.count = 0

            self.env = gym.make(game)

            s_t = self.resize_image(self.env.reset())

            self.s_t = np.tile(s_t, [1, 1, frames])
            self.s = [self.s_t]

            self.a_t = random.randint(0, act_space - 1)
            self.a = [self.a_t]
            self.a_logits = []
            self.r = [0]

            self.v_cur = []

            state_in = np.zeros(self.state_size, dtype=np.float32)
            self.state_in = [state_in]

            self.done = False

        def step(self, a, a_logits, v_cur, state_in, force=False):
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
            s_t1 = self.resize_image(s_t1)
            channels = s_t1.shape[-1]
            self.s_t = np.concatenate([s_t1, self.s_t[:, :, :-channels]], axis=-1)

            self.s.append(self.s_t)
            self.a.append(self.a_t)
            self.a_logits.append(a_logits)
            self.r.append(r_t)
            self.done = done

            self.v_cur.append(v_cur)
            self.state_in.append(state_in)

            """
            get segs
            """
            segs = self.get_history(force)

            """
            reset env
            """
            self.reset(force)

            return segs

        def reset(self, force=False):
            if self.done or force:
                print("  Total Reward  %s : %d" % (self.game, np.sum(self.r)))
                self.count = 0
                self.act_repeat = random.choice(self.act_repeats)

                s_t = self.resize_image(self.env.reset())

                self.s_t = np.tile(s_t, [1, 1, self.frames])
                self.s = [self.s_t]

                self.a_t = random.randint(0, self.act_space - 1)
                self.a = [self.a_t]
                self.a_logits = []
                self.r = [0]

                self.v_cur = []

                state_in = np.zeros(self.state_size, dtype=np.float32)
                self.state_in = [state_in]

                self.done = False

        def get_state(self):
            return self.s_t

        def get_act(self):
            return self.a_t

        def get_state_in(self):
            return self.state_in[-1]

        def get_history(self, force=False):
            if self.done or force:
                if self.done:
                    gaes = get_gaes(None, self.r, self.v_cur, self.v_cur[1:] + [0], 0.99, 0.95)[0]
                    seg = Seg(self.s, self.a, self.a_logits, self.r, gaes, self.v_cur, self.state_in)
                    return self.postprocess(seg)
                if force and len(self.r) > 1:
                    gaes = get_gaes(None, self.r[:-1], self.v_cur[:-1], self.v_cur[1:], 0.99, 0.95)[0]
                    seg = Seg(self.s[:-1], self.a[:-1], self.a_logits[:-1], self.r[:-1], gaes,
                              self.v_cur[:-1], self.state_in[:-1])
                    return self.postprocess(seg)
            return None

        @staticmethod
        def resize_image(image, size=84):
            image = Image.fromarray(image)
            image = image.resize((size, size))
            image = np.array(image)
            image = image / 255.
            image = np.array(image, np.float32)
            return image

        def postprocess(self, seg):
            """
            postprocess the seg for training
            :author lhw
            """
            burn_in = self.burn_in
            seqlen = self.seqlen + burn_in
            seg_results = []
            if seg is not None:
                while len(seg[0]) > burn_in:
                    next_seg = dict()
                    next_seg["s"] = padding(seg.s[:seqlen], seqlen, np.float32)
                    next_seg["a"] = padding(seg.a[1:seqlen + 1], seqlen, np.int32)
                    next_seg["prev_a"] = padding(seg.a[:seqlen], seqlen, np.int32)
                    next_seg["a_logits"] = padding(seg.a_logits[:seqlen], seqlen, np.float32)
                    next_seg["r"] = padding(seg.r[1:seqlen + 1], seqlen, np.float32)
                    next_seg["prev_r"] = padding(seg.r[:seqlen], seqlen, np.float32)
                    next_seg["adv"] = padding(seg.gaes[:seqlen], seqlen, np.float32)
                    next_seg["v_cur"] = padding(seg.v_cur[:seqlen], seqlen, np.float32)
                    next_seg["state_in"] = np.array(seg.state_in[0], np.float32)
                    next_seg["slots"] = padding(len(seg.s[:seqlen]) * [1], seqlen, np.int32)

                    seg_results.append(next_seg)
                    seg = Seg(*[t[burn_in:] for t in seg])
            if any(seg_results):
                # print("full use one segs done!")
                return seg_results
            else:
                return None

    class Envs(object):
        def __init__(self, act_space, act_repeats, frames,
                     state_size, burn_in, seqlen, games):
            self.envs = []
            for game in games:
                env = Env(act_space, act_repeats, frames,
                          state_size, burn_in, seqlen, game)
                self.envs.append(env)

        def step(self, sess, model):
            fd = self.get_feed_dict(model)

            a, a_logits, v_cur, state_in = sess.run(
                [model.current_act, model.current_act_logits,
                 model.current_value, model.state_out],
                feed_dict=fd)

            segs = [env.step(
                a[i][0],
                a_logits[i][0],
                v_cur[i][0],
                state_in[i]
            ) for (i, env) in enumerate(self.envs)]

            segs = [t2 for t1 in segs if t1 is not None for t2 in t1]

            return segs

        def get_feed_dict(self, model):
            fd = dict()
            fd[model.s_t] = [[env.get_state()] for env in self.envs]
            fd[model.previous_actions] = [[env.get_act()] for env in self.envs]
            fd[model.prev_r] = [[env.r[-1]] for env in self.envs]
            fd[model.state_in] = [env.get_state_in() for env in self.envs]
            return fd

    return Envs


def build_env(kwargs):
    Envs = _warp_env()
    state_size = kwargs['state_size']
    action_repeats = kwargs['action_repeats']
    frames = kwargs["frames"]
    parallel = kwargs['parallel']
    act_space = kwargs['act_space']
    burn_in = kwargs['burn_in']
    seqlen = kwargs['seqlen']

    games = ["DeepmindLabContributedDmlab30ExploreGoalLocationsLarge-v0"]
    games = games * (parallel // len(games))

    envs = Envs(act_space, action_repeats, frames,
                state_size, burn_in, seqlen, games)

    return envs
