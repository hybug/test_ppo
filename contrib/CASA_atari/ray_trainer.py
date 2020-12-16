# coding: utf-8

import ray
import sys

sys.path.append("/opt/tiger/test_ppo")

from ray_helper.rollout_collector import RolloutCollector
from ray_helper.asyncps import AsyncPS
from ray_helper.rollout_collector import QueueReader, fetch_one_structure
from ray_helper.miscellaneous import init_cluster_ray, warp_mkdir, warp_exists

import logging
import os
import time
import pickle
import numpy as np
from collections import namedtuple

import gym
import tensorflow as tf

from algorithm import dPPOcC
from train_ops import miniOp
from train_ops import assignOp
from module import TmpHierRNN
from module import RMCRNN
from module import AMCRNN
from module import mse
from module import KL_from_gaussians
from module import entropy_from_logits
from utils import get_shape
from utils import PrioritizedReplayBuffer
from utils import ReplayBuffer

from contrib.CASA_atari.policy_graph import warp_Model
from contrib.CASA_atari.game_list import get_games

logging.getLogger('tensorflow').setLevel(logging.ERROR)

NEST = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")
flags.DEFINE_integer("game", 0, "game")

flags.DEFINE_string(
    "basedir_ceph", "/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
                    "/CASA_atari", "base dir for ceph")
flags.DEFINE_string(
    "basedir_hdfs",
    'hdfs://haruna/home/byte_arnold_lq_mlsys/user/xiaochangnan/atari',
    "base dir for hdfs")
flags.DEFINE_string("dir", "casa0", "dir number")
flags.DEFINE_string(
    "scriptdir",
    "/opt/tiger/test_ppo/contrib/CASA_atari",
    "script dir")

flags.DEFINE_bool("use_stage", False, "whether to use tf.contrib.staging")

flags.DEFINE_integer("use_hrnn", 0, "whether to use tmp hierarchy rnn (lstm+rmc)")
flags.DEFINE_integer("use_rmc", 0, "whether to use rmc")
flags.DEFINE_integer("use_amc", 0, "whether to use amc")

flags.DEFINE_integer("use_beta", 1, "whether to use beta")
flags.DEFINE_integer("use_reward_prediction", 1, "whether to use reward prediction")
flags.DEFINE_integer("after_rnn", 1, "whether to use reward prediction after rnn")
flags.DEFINE_integer("use_pixel_control", 1, "whether to use pixel control")
flags.DEFINE_integer("use_retrace", 1, "whether to use retrace to estimate q")
flags.DEFINE_integer("smooth_update", 1, "whether to smooth update q target")

flags.DEFINE_float("pq_kl_coef", 0.1, "weight of kl between posterior and prior")
flags.DEFINE_float("p_kl_coef", 0.01, "weight of kl between prior and normal gaussian")

flags.DEFINE_bool("use_hdfs", True, "whether to use hdfs")

flags.DEFINE_integer("seqlen", 40, "seqlen of each training segment")
flags.DEFINE_integer("burn_in", 40, "seqlen of each burn-in segment")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("total_environment_frames", 1000000000,
                     "total num of frames for train")

flags.DEFINE_float("init_lr", 1e-3, "initial learning rate")
flags.DEFINE_float("lr_decay", 1.0, "whether to decay learning rate")
flags.DEFINE_float("warmup_steps", 4000, "steps for warmup")

flags.DEFINE_integer("frames", 1, "num of channels of each state")
flags.DEFINE_integer("image_size", 84, "input image size")
flags.DEFINE_float("gamma", 0.99, "discount rate")
flags.DEFINE_integer("time_scale", 4, "time scale of hierarchy rnn")
flags.DEFINE_integer("n_step", 5, "n_step return")
flags.DEFINE_float("vf_clip", 1.0, "clip of value function")
flags.DEFINE_float("ppo_clip", 0.2, "clip of ppo loss")
flags.DEFINE_float("pi_coef", 20.0, "weight of policy fn loss")
flags.DEFINE_float("vf_coef", 1.0, "weight of v-value fn loss")
flags.DEFINE_float("qf_coef", 1.0, "weight of q-value fn loss")
flags.DEFINE_float("ent_coef", 1.0, "weight of entropy loss")
flags.DEFINE_integer("target_update", 1000, "target net update per steps")
flags.DEFINE_bool("zero_init", False, "whether to zero init initial state")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")
flags.DEFINE_integer("replay", 1, "times of replay")

flags.DEFINE_integer("seed", 12358, "random seed")

# param for ray evaluator
flags.DEFINE_float("timeout", 0.1, "get operation timeout")
flags.DEFINE_integer("num_returns", 4, "nof data of wait operation")
flags.DEFINE_integer('load_ckpt_period', 800, 'for how many step to load ckpt in inf server')
flags.DEFINE_integer('qsize', 10, 'for how many qsize * batchsize in main procress')
flags.DEFINE_integer('nof_server_gpus', 1, 'nof_gpus for training')
flags.DEFINE_integer('cpu_per_actor', 2, 'nof cpu required for infserver, -1 for not require')
flags.DEFINE_integer('nof_evaluator', 1, 'nof_gpus for training')

Model = warp_Model()


def build_learner(pre, post, act_space, num_frames, batch_weights):
    global_step = tf.train.get_or_create_global_step()
    init_lr = FLAGS.init_lr
    decay = FLAGS.lr_decay
    warmup_steps = FLAGS.warmup_steps
    gamma = FLAGS.gamma
    n_step = FLAGS.n_step
    time_scale = FLAGS.time_scale
    use_hrnn = FLAGS.use_hrnn
    use_rmc = FLAGS.use_rmc
    use_amc = FLAGS.use_amc
    use_beta = FLAGS.use_beta
    use_retrace = FLAGS.use_retrace
    use_reward_prediction = FLAGS.use_reward_prediction
    after_rnn = FLAGS.after_rnn
    use_pixel_control = FLAGS.use_pixel_control
    pq_kl_coef = FLAGS.pq_kl_coef
    p_kl_coef = FLAGS.p_kl_coef
    pi_coef = FLAGS.pi_coef
    vf_coef = FLAGS.vf_coef
    ent_coef = FLAGS.ent_coef
    qf_coef = FLAGS.qf_coef
    ppo_clip = FLAGS.ppo_clip
    vf_clip = FLAGS.vf_clip

    global_step_float = tf.cast(global_step, tf.float32)

    lr = tf.train.polynomial_decay(
        init_lr, global_step,
        FLAGS.total_environment_frames // (
                FLAGS.batch_size * FLAGS.seqlen),
        init_lr / 10.)
    is_warmup = tf.cast(global_step_float < warmup_steps, tf.float32)
    lr = is_warmup * global_step_float / warmup_steps * init_lr + (
            1.0 - is_warmup) * (init_lr * (1.0 - decay) + lr * decay)

    ent_coef = tf.train.polynomial_decay(
        ent_coef, global_step,
        FLAGS.total_environment_frames // (
                FLAGS.batch_size * FLAGS.seqlen),
        ent_coef / 10.,
        power=1.0)

    optimizer = tf.train.AdamOptimizer(lr)

    if FLAGS.zero_init:
        pre["state_in"] = tf.zeros_like(pre["state_in"])

    if use_hrnn:
        rnn = TmpHierRNN(
            time_scale, 64, 4, 2, 8, 'lstm', 'rmc',
            return_sequences=True, return_state=True, name="hrnn")
    elif use_rmc:
        rnn = RMCRNN(
            64, 4, 64,
            return_sequences=True, return_state=True, name="rmc")
    elif use_amc:
        rnn = AMCRNN(
            64, 4, 64,
            return_sequences=True, return_state=True, name="amc")
    else:
        rnn = tf.compat.v1.keras.layers.CuDNNLSTM(
            256, return_sequences=True, return_state=True, name="lstm")

    pre_model = Model(
        act_space, gamma, n_step, rnn, use_hrnn, use_rmc, use_amc, use_beta, use_reward_prediction,
        after_rnn, use_pixel_control, False, **pre)

    post["state_in"] = tf.stop_gradient(pre_model.state_out)

    post_model = Model(
        act_space, gamma, n_step, rnn, use_hrnn, use_rmc, use_amc, use_beta, use_reward_prediction,
        after_rnn, use_pixel_control, True, **post)

    tf.summary.scalar("adv_mean", post_model.adv_mean)
    tf.summary.scalar("adv_std", post_model.adv_std)

    if use_retrace:
        q_loss = mse(post_model.qa, post_model.retrace_qs)
    else:
        q_loss = mse(post_model.qa, post_model.n_step_qs)
    # q_loss = mse(
    #     post_model.qa,
    #     tf.stop_gradient(
    #         post_model.current_value[:, :-n_step] + post_model.adv))
    q_loss = tf.reduce_mean(
        q_loss * post_model.mask[:, :-n_step] * batch_weights[:, None]
    ) + 3.0 * tf.reduce_mean(
        q_loss * post_model.mask[:, :-n_step] * (1.0 - batch_weights[:, None]))

    ent_loss = tf.reduce_mean(
        entropy_from_logits(
            post_model.current_act_logits
        ) * post_model.mask * batch_weights[:, None])

    losses = dPPOcC(act=post_model.a[:, 1:1 - n_step],
                    policy_logits=post_model.current_act_logits[:, :-n_step, :],
                    behavior_logits=post_model.behavior_logits[:, :-n_step, :],
                    advantage=post_model.adv,
                    policy_clip=ppo_clip,
                    vf=post_model.current_value[:, :-n_step],
                    vf_target=post_model.vs,
                    value_clip=vf_clip,
                    old_vf=post_model.old_vf[:, :-n_step])
    p_loss = tf.reduce_mean(losses.p_loss * post_model.mask[:, :-n_step] * batch_weights[:, None])
    v_loss = tf.reduce_mean(losses.v_loss * post_model.mask[:, :-n_step] * batch_weights[:, None])

    add_loss = 0.0
    if use_hrnn:
        pq_kl_loss = KL_from_gaussians(
            post_model.q_mus, post_model.q_sigmas,
            post_model.p_mus, post_model.p_sigmas)
        pq_kl_loss = tf.reduce_mean(pq_kl_loss * post_model.mask)

        p_kl_loss = KL_from_gaussians(
            post_model.p_mus, post_model.p_sigmas,
            tf.zeros_like(post_model.p_mus), 0.01 * tf.ones_like(post_model.p_sigmas))
        p_kl_loss = tf.reduce_mean(p_kl_loss * post_model.mask)

        with tf.name_scope("hierarchy_loss"):
            tf.summary.scalar("kl_div_pq", pq_kl_loss)
            tf.summary.scalar("kl_div_prior", p_kl_loss)
        add_loss += pq_kl_coef * pq_kl_loss
        add_loss += p_kl_coef * p_kl_loss

    if use_reward_prediction:
        r_loss = tf.reduce_mean(mse(
            post_model.reward_prediction,
            post_model.r[:, 1: 1 - n_step]
        ) * post_model.mask[:, :-n_step])
        tf.summary.scalar("r_loss", r_loss)
        add_loss += r_loss

    if use_pixel_control:
        s = tf.cast(post_model.s[:, : 1 - n_step, :, :, :], tf.float32) / 255.0
        target = s[:, 1:, :, :, :] - s[:, :-1, :, :, :]
        shape = get_shape(target)
        target = tf.reshape(
            target,
            (shape[0], shape[1], 4, shape[2] // 4, 4, shape[3] // 4, shape[4]))
        target = tf.reduce_mean(target, axis=(2, 4))
        pixel_loss = tf.reduce_mean(mse(
            post_model.pixel_control,
            target
        ) * post_model.mask[:, :-n_step, None, None, None])
        with tf.name_scope("control_loss"):
            tf.summary.scalar("pixel_control_loss", pixel_loss)
        add_loss += 100.0 * pixel_loss

    loss = (qf_coef * q_loss
            + vf_coef * v_loss
            + pi_coef * p_loss
            - ent_coef * ent_loss
            + add_loss)

    abs_td = post_model.mask[:, :-n_step] * tf.abs(
        post_model.qa - post_model.n_step_rewards
        + gamma ** n_step * post_model.qa1)
    avg_p = tf.reduce_mean(abs_td, axis=-1)
    max_p = tf.reduce_max(abs_td, axis=-1)
    priority = 0.9 * max_p + 0.1 * avg_p

    beta = tf.train.polynomial_decay(
        0.4, global_step,
        FLAGS.total_environment_frames // (
                FLAGS.batch_size * FLAGS.seqlen),
        1.0)

    train_op = miniOp(optimizer, loss, FLAGS.grad_clip)

    if FLAGS.smooth_update:
        init_target_op = assignOp(
            1.0,
            {"q": "q_target"})
        target_op = assignOp(
            1.0 / FLAGS.target_update,
            {"q": "q_target"})
    else:
        init_target_op = assignOp(
            1.0,
            {"q": "q_target"})
        target_op = tf.no_op()

    dependency = [train_op, target_op]

    new_frames = tf.reduce_sum(post["mask"])

    with tf.control_dependencies(dependency):
        num_frames_and_train = tf.assign_add(
            num_frames, new_frames)
        global_step_and_train = tf.assign_add(
            global_step, 1)

    tf.summary.scalar("learning_rate", lr)
    tf.summary.scalar("pi_loss", p_loss)
    tf.summary.scalar("q_loss", q_loss)
    tf.summary.scalar("v_loss", v_loss)
    tf.summary.scalar("ent_loss", ent_loss)
    tf.summary.scalar("all_loss", loss)

    return num_frames_and_train, global_step_and_train, init_target_op, priority, beta


def build_policy_evaluator(kwargs):
    """
    construct policy_evaluator
    :param kwargs:
    :return: construcation method and params for Evaluator
    """
    from copy import deepcopy

    if kwargs["use_hrnn"]:
        kwargs["state_size"] = 1 + (8 + 2 + 8) * 4 * 64
    elif kwargs["use_rmc"]:
        kwargs["state_size"] = 64 * 4 * 64
    elif kwargs["use_amc"]:
        kwargs["state_size"] = 1 + 64 + 64 + 64 * 4 * 64
    else:
        kwargs['state_size'] = 256 * 2

    env_kwargs = deepcopy(kwargs)
    env_kwargs['action_repeats'] = [1]

    model_kwargs = deepcopy(kwargs)
    # pickle func in func
    from contrib.CASA_atari.env import build_env
    from contrib.CASA_atari.policy_graph import build_evaluator_model
    return model_kwargs, build_evaluator_model, env_kwargs, build_env


def init_dir_and_log():
    tf.set_random_seed(FLAGS.seed)
    if FLAGS.use_hdfs:
        base_dir = FLAGS.basedir_hdfs
    else:
        base_dir = FLAGS.basedir_ceph
    games = get_games()
    base_dir = os.path.join(base_dir, FLAGS.dir, games[FLAGS.game])

    if not warp_exists(base_dir, use_hdfs=FLAGS.use_hdfs):
        warp_mkdir(base_dir, FLAGS.use_hdfs)

    ckpt_dir = os.path.join(base_dir, "ckpt")
    if not warp_exists(ckpt_dir, FLAGS.use_hdfs):
        warp_mkdir(ckpt_dir, FLAGS.use_hdfs)

    local_log_dir = os.path.join(FLAGS.scriptdir, 'log')
    if not os.path.exists(local_log_dir):
        os.mkdir(local_log_dir)
    logging.basicConfig(filename=os.path.join(
        local_log_dir, "Trainerlog"), level="INFO")

    summary_dir = os.path.join(base_dir, "summary")
    if not warp_exists(summary_dir, FLAGS.use_hdfs):
        warp_mkdir(summary_dir, FLAGS.use_hdfs)
    return base_dir, ckpt_dir, summary_dir


def train():
    """
    init dir and log config
    """
    init_cluster_ray()
    base_dir, ckpt_dir, summary_dir = init_dir_and_log()

    kwargs = FLAGS.flag_values_dict()
    kwargs["BASE_DIR"] = base_dir
    kwargs["ckpt_dir"] = ckpt_dir
    games = get_games()
    kwargs["game"] = games[kwargs["game"]]
    env = gym.make(kwargs["game"])
    act_space = env.action_space.n
    kwargs["act_space"] = act_space
    del env
    """
    get one seg from rollout worker for dtype and shapes

    :param kwargs rollout worker config
    """
    logging.info('get one seg from Evaluator for dtype and shapes')
    ps = AsyncPS.remote()
    small_data_collector = RolloutCollector(
        server_nums=1, ps=ps,
        policy_evaluator_build_func=build_policy_evaluator,
        **kwargs)
    cache_struct_path = '/tmp/%s_%s.pkl' % (FLAGS.dir, kwargs["game"])
    structure = fetch_one_structure(
        small_data_collector,
        cache_struct_path=cache_struct_path, is_head=True)
    del small_data_collector

    """
        init data prefetch thread, prepare_input_pipe
    """
    keys = list(structure.keys())
    dtypes = [structure[k].dtype for k in keys]
    shapes = [structure[k].shape for k in keys]
    segBuffer = tf.queue.RandomShuffleQueue(
        capacity=FLAGS.qsize * FLAGS.batch_size,
        min_after_dequeue=FLAGS.qsize * FLAGS.batch_size // 2,
        dtypes=dtypes,
        shapes=shapes,
        names=keys,
        shared_name="buffer")
    server_nums = FLAGS.nof_evaluator
    nof_server_gpus = FLAGS.nof_server_gpus
    server_nums_refine = server_nums // nof_server_gpus
    data_collector = RolloutCollector(
        server_nums=server_nums_refine, ps=ps,
        policy_evaluator_build_func=build_policy_evaluator, **kwargs)

    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=1))
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    reader = QueueReader(
        sess=sess,
        global_queue=segBuffer,
        data_collector=data_collector,
        keys=keys,
        dtypes=dtypes,
        shapes=shapes)
    reader.daemon = True
    reader.start()

    dequeued = segBuffer.dequeue_many(FLAGS.batch_size)

    # //////////////////////
    from_where = dequeued
    batch_weights = tf.ones(FLAGS.batch_size)

    # //////////////////////

    prephs, postphs = dict(), dict()
    for k, v in from_where.items():
        if k == "state_in":
            prephs[k] = v
        else:
            prephs[k], postphs[k] = tf.split(
                v, [FLAGS.burn_in, FLAGS.seqlen + FLAGS.n_step], axis=1)
    prekeys = list(prephs.keys())
    postkeys = list(postphs.keys())

    """
        count frame and total steps
    """
    num_frames = tf.get_variable(
        'num_environment_frames',
        initializer=tf.zeros_initializer(),
        shape=[],
        dtype=tf.int32,
        trainable=False)
    tf.summary.scalar("frames", num_frames)
    global_step = tf.train.get_or_create_global_step()

    dur_time_tensor = tf.placeholder(dtype=tf.float32)
    tf.summary.scalar('time_per_step', dur_time_tensor)

    """
        set stage_op and build learner
    """
    with tf.device("/gpu"):
        if FLAGS.use_stage:
            area = tf.contrib.staging.StagingArea(
                [prephs[key].dtype for key in prekeys] + [postphs[key].dtype for key in postkeys],
                [prephs[key].shape for key in prekeys] + [postphs[key].shape for key in postkeys])
            stage_op = area.put([prephs[key] for key in prekeys] + [postphs[key] for key in postkeys])
            from_stage = area.get()
            predatas = {key: from_stage[i] for i, key in enumerate(prekeys)}
            postdatas = {key: from_stage[i + len(prekeys)] for i, key in enumerate(postkeys)}
        else:
            stage_op = []
            predatas, postdatas = prephs, postphs

        num_frames_and_train, global_step_and_train, init_target_op, priority, beta = build_learner(
            pre=predatas, post=postdatas, act_space=act_space, num_frames=num_frames, batch_weights=batch_weights)

    """
        add summary
    """
    summary_ops = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    """
        initialize and save ckpt
    """
    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=6)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    ws = Model.get_ws(sess)
    logging.info('pushing weight to ps')
    ray.get(ps.push.remote(ws))

    saver.save(sess, os.path.join(ckpt_dir, "CKPT"), global_step=global_step)

    """
        step
    """
    total_frames = 0
    sess.run(stage_op)
    sess.run(init_target_op)
    dur_time = 0
    while total_frames < FLAGS.total_environment_frames:
        start = time.time()

        fd = {dur_time_tensor: dur_time}

        total_frames, gs, summary, _ = sess.run(
            [num_frames_and_train, global_step_and_train,
             summary_ops, stage_op],
            feed_dict=fd)

        if gs % FLAGS.target_update == 0:
            sess.run(init_target_op)

        if gs % 25 == 0:
            ws = Model.get_ws(sess)
            logging.info('pushing weight to ps')
            try:
                ray.get(ps.push.remote(ws))
            except ray.exceptions.UnreconstructableError as e:
                logging.info(str(e))
            except ray.exceptions.RayError as e:
                logging.info(str(e))

        if gs % 1000 == 0:
            saver.save(sess, os.path.join(ckpt_dir, "CKPT"), global_step=global_step)

        if gs % 1 == 0:
            summary_writer.add_summary(summary, global_step=gs)
            dur_time = time.time() - start
            msg = "Global Step %d, Total Frames %d,  Time Consume %.2f" % (
                gs, total_frames, dur_time)
            logging.info(msg)

    saver.save(sess, os.path.join(ckpt_dir, "CKPT"), global_step=global_step)

    nof_workers = os.getenv('ARNOLD_WORKER_NUM', None)
    assert nof_workers is not None, nof_workers

    for i in range(int(nof_workers)):
        print('killing worker %s' % i)
        os.system('ssh worker-%s pkill run.sh' % i)


def main(_):
    if FLAGS.mode == "train":
        train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
