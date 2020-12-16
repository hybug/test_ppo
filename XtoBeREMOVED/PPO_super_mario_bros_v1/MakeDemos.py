# coding: utf-8

import tensorflow as tf
import os
import pyarrow as pa

from utils import pack

from XtoBeREMOVED.PPO_super_mario_bros_v1.Trainer_PPO import Model
from XtoBeREMOVED.PPO_super_mario_bros_v1.Test_PPO import Env

CKPT_DIR = "../ckpt/8"

frames = 1
action_repeats = [1]
MAX_STEPS = 320000
seqlen = 64
CLIP = 1.0

sess = tf.Session()

phs = dict()

phs["s"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, frames])
phs["prev_a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
phs["a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
phs["a_logits"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 7])
phs["adv"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
phs["v_cur"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 256 * 2])
phs["slots"] = tf.placeholder(dtype=tf.float32, shape=[None, None])

lstm = tf.compat.v1.keras.layers.LSTM(
    256, return_sequences=True, return_state=True, name="lstm")
model = Model(7, lstm, "agent", **phs)

saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=6)

ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
saver.restore(sess, os.path.join(CKPT_DIR, ckpt.model_checkpoint_path.split("/")[-1]))

game = "SuperMarioBros-8-3-v0"
env = Env(7, action_repeats, frames, game)

count = 1


def step(e=None):
    _s_t_batch = env.get_state()[None, None, :, :, :]
    _a_t_batch = [[env.get_act()]]
    _state_in_batch = [env.get_state_in()]

    _a_t_new, _a_t_logits, _v_cur, _state_out_batch = sess.run(
        [model.get_current_act(),
         model.get_current_act_logits(),
         model.current_value,
         model.state_out],
        feed_dict={model.s_t: _s_t_batch,
                   model.previous_actions: _a_t_batch,
                   model.state_in: _state_in_batch})

    # _a_t_new = np.argmax(_a_t_logits, axis=-1)

    if e is not None:
        _a_t_new[0][0] = e

    env.step(
        _a_t_new[0][0],
        _a_t_logits[0][0],
        _state_out_batch[0]
    )

    env.update_v(_v_cur[0][0])

    seg = env.get_history()
    if seg is not None:
        if not os.path.exists("Demos"):
            os.mkdir("Demos")
        dicseg = {k: v for k, v in seg._asdict().items()}
        with pa.OSFile(os.path.join("Demos", game + "_%d.demo" % count), "wb") as f:
            f.write(pack(dicseg))
    env.reset()


if __name__ == '__main__':
    pass
