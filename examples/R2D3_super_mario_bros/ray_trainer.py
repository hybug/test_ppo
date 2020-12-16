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
from collections import namedtuple
import numpy as np
import tensorflow as tf

from train_ops import miniOp
from train_ops import assignOp
from module import TmpHierRNN
from module import mse
from module import KL_from_gaussians
from module import rescaleTarget
from module import entropy_from_logits
from utils import get_shape
from utils import PrioritizedReplayBuffer

logging.getLogger('tensorflow').setLevel(logging.ERROR)

NEST = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")
flags.DEFINE_string("act_space", "12", "act space")

flags.DEFINE_string(
    "basedir_ceph", "/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
                    "/PPOcGAE_SuperMarioBros-v0", "base dir for ceph")
flags.DEFINE_string(
    "basedir_hdfs",
    'hdfs://haruna/home/byte_arnold_lq_mlsys/user/xiaochangnan/test_ppo',
    "base dir for hdfs")
flags.DEFINE_string("dir", "R2D3_0", "dir number")
flags.DEFINE_string(
    "scriptdir",
    "/opt/tiger/test_ppo/examples/R2D3_super_mario_bros",
    "script dir")

flags.DEFINE_bool("use_stage", False, "whether to use tf.contrib.staging")

flags.DEFINE_integer("use_soft", 1, "whether to use soft")
flags.DEFINE_integer("use_hrnn", 1, "whether to use tmp hierarchy rnn (lstm+rmc)")
flags.DEFINE_integer("use_reward_prediction", 1, "whether to use reward prediction")
flags.DEFINE_integer("after_rnn", 1, "whether to use reward prediction after rnn")
flags.DEFINE_integer("use_pixel_control", 1, "whether to use pixel control")
flags.DEFINE_integer("sample_epsilon_per_step", 0, "whether to sample epsilon per step or per trajectory")

flags.DEFINE_float("pq_kl_coef", 0.1, "weight of kl between posterior and prior")
flags.DEFINE_float("p_kl_coef", 0.01, "weight of kl between prior and normal gaussian")

flags.DEFINE_bool("use_hdfs", True, "whether to use hdfs")

flags.DEFINE_integer("seqlen", 32, "seqlen of each training segment")
flags.DEFINE_integer("burn_in", 32, "seqlen of each burn-in segment")
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
flags.DEFINE_float("vf_coef", 100.0, "weight of value fn loss")
flags.DEFINE_integer("target_update", 2500, "target net update per steps")
flags.DEFINE_bool("zero_init", False, "whether to zero init initial state")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")

flags.DEFINE_integer("seed", 12358, "random seed")

# param for ray evaluator
flags.DEFINE_float("timeout", 0.1, "get operation timeout")
flags.DEFINE_integer("num_returns", 8, "nof data of wait operation")
flags.DEFINE_integer('load_ckpt_period', 10, 'for how many step to load ckpt in inf server')
flags.DEFINE_integer('qsize', 8, 'for how many qsize * batchsize in main procress')
flags.DEFINE_integer('nof_server_gpus', 1, 'nof_gpus for training')
flags.DEFINE_integer('cpu_per_actor', 2, 'nof cpu required for infserver, -1 for not require')
flags.DEFINE_integer('nof_evaluator', 1, 'nof_gpus for training')

from examples.R2D3_super_mario_bros.policy_graph import warp_Model
from examples.R2D3_super_mario_bros.env import padding

Model = warp_Model()


def build_demo_buffer(keys, alpha):
    seg_results = []
    games = ["SuperMarioBros-%d-%d-v0" % (i, j) for i in range(1, 9) for j in range(1, 5)]

    if FLAGS.use_hrnn:
        state_size = 1 + (8 + 2 + 8) * 4 * 64
    else:
        state_size = 256 * 2

    Demo = namedtuple("Demo", ["s", "a", "r", "mask"])

    def postprocess_one_seg(seg):
        seqlen = FLAGS.seqlen + FLAGS.burn_in + FLAGS.n_step

        next_seg = dict()

        next_seg["s"] = padding(seg.s[:seqlen], seqlen, np.uint8)
        next_seg["a"] = padding(seg.a[:seqlen], seqlen, np.int32)
        next_seg["r"] = padding(seg.r[:seqlen], seqlen, np.float32)
        next_seg["state_in"] = np.zeros(state_size, np.float32)
        next_seg["mask"] = padding(seg.mask[:seqlen], seqlen, np.int32)

        return next_seg

    for game in games:
        with open("/opt/tiger/test_ppo/Demos/Demo_%s.pkl" % game, "rb") as f:
            seg = pickle.load(f)
        seg = Demo(seg["s"], seg["a"], seg["r"], np.ones(len(seg["s"])))
        while len(seg[0]) > FLAGS.burn_in + FLAGS.n_step:
            next_seg = postprocess_one_seg(seg)
            seg_results.append(next_seg)
            seg = Demo(*[t[1:] for t in seg])

    demo_buffer = PrioritizedReplayBuffer(len(seg_results), keys, alpha)
    demo_buffer._max_priority = 10.0
    for seg in seg_results:
        demo_buffer.add(seg)
    return demo_buffer


def build_learner(pre, post, ws, act_space, num_frames):
    global_step = tf.train.get_or_create_global_step()
    init_lr = FLAGS.init_lr
    decay = FLAGS.lr_decay
    warmup_steps = FLAGS.warmup_steps
    gamma = FLAGS.gamma
    n_step = FLAGS.n_step
    use_soft = FLAGS.use_soft
    time_scale = FLAGS.time_scale
    use_hrnn = FLAGS.use_hrnn
    use_reward_prediction = FLAGS.use_reward_prediction
    after_rnn = FLAGS.after_rnn
    use_pixel_control = FLAGS.use_pixel_control
    pq_kl_coef = FLAGS.pq_kl_coef
    p_kl_coef = FLAGS.p_kl_coef

    global_step_float = tf.cast(global_step, tf.float32)

    lr = tf.train.polynomial_decay(
        init_lr, global_step,
        FLAGS.total_environment_frames // (
                FLAGS.batch_size * FLAGS.seqlen),
        init_lr / 10.)
    is_warmup = tf.cast(global_step_float < warmup_steps, tf.float32)
    lr = is_warmup * global_step_float / warmup_steps * init_lr + (
            1.0 - is_warmup) * (init_lr * (1.0 - decay) + lr * decay)
    optimizer = tf.train.AdamOptimizer(lr)

    if FLAGS.zero_init:
        pre["state_in"] = tf.zeros_like(pre["state_in"])

    if use_hrnn:
        rnn = TmpHierRNN(
            time_scale, 64, 4, 2, 8, 'lstm', 'rmc',
            return_sequences=True, return_state=True, name="hrnn")
    else:
        rnn = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")

    pre_model = Model(
        act_space, gamma, n_step, use_soft, rnn, use_hrnn, use_reward_prediction,
        after_rnn, use_pixel_control, False, **pre)

    post["state_in"] = tf.stop_gradient(pre_model.state_out)

    post_model = Model(
        act_space, gamma, n_step, use_soft, rnn, use_hrnn, use_reward_prediction,
        after_rnn, use_pixel_control, True, **post)

    v_loss = mse(post_model.qa,
                 tf.stop_gradient(
                     rescaleTarget(post_model.n_step_rewards,
                                   gamma ** n_step,
                                   post_model.qa1)))
    v_loss = FLAGS.vf_coef * tf.reduce_mean(
        v_loss * post_model.mask[:, :-n_step] * ws[:, None])

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
        add_loss += pixel_loss

    loss = FLAGS.vf_coef * v_loss + add_loss

    abs_td = post_model.mask[:, :-n_step] * tf.abs(
        post_model.qa - rescaleTarget(
            post_model.n_step_rewards,
            gamma ** n_step,
            post_model.qa1))
    avg_p = tf.reduce_mean(abs_td, axis=-1)
    max_p = tf.reduce_max(abs_td, axis=-1)
    priority = 0.9 * max_p + 0.1 * avg_p

    beta = tf.train.polynomial_decay(
        0.4, global_step,
        FLAGS.total_environment_frames // (
                FLAGS.batch_size * FLAGS.seqlen),
        1.0)

    train_op = miniOp(optimizer, loss, FLAGS.grad_clip)

    target_op = assignOp(
        1.0,
        {"q": "q_target"})

    dependency = [train_op]
    if use_soft:
        qf_entropy = entropy_from_logits(post_model.qf_logits)
        target_entropy = tf.train.polynomial_decay(
            0.9 * np.log(act_space), global_step,
            FLAGS.total_environment_frames // (
                    FLAGS.batch_size * FLAGS.seqlen),
            0.5 * np.log(act_space))
        ent_loss = tf.reduce_mean(
            mse(qf_entropy,
                tf.cast(target_entropy, tf.float32)[None, None]))
        with tf.name_scope("ent_loss"):
            tf.summary.scalar("ent_loss", ent_loss)
        ent_op = miniOp(
            optimizer, ent_loss,
            grad_clip=FLAGS.grad_clip, var_scope="temperature")
        dependency.append(ent_op)

    new_frames = tf.reduce_sum(post["mask"])

    with tf.control_dependencies(dependency):
        num_frames_and_train = tf.assign_add(
            num_frames, new_frames)
        global_step_and_train = tf.assign_add(
            global_step, 1)

    tf.summary.scalar("learning_rate", lr)
    tf.summary.scalar("v_loss", v_loss)
    tf.summary.scalar("all_loss", loss)

    return num_frames_and_train, global_step_and_train, target_op, priority, beta


def build_policy_evaluator(kwargs):
    """
    construct policy_evaluator
    :param kwargs:
    :return: construcation method and params for Evaluator
    """
    from copy import deepcopy

    if kwargs["use_hrnn"]:
        kwargs["state_size"] = 1 + (8 + 2 + 8) * 4 * 64
    else:
        kwargs['state_size'] = 256 * 2

    env_kwargs = deepcopy(kwargs)
    env_kwargs['action_repeats'] = [1]

    model_kwargs = deepcopy(kwargs)
    # pickle func in func
    from examples.R2D3_super_mario_bros.env import build_env
    from examples.R2D3_super_mario_bros.policy_graph import build_evaluator_model
    return model_kwargs, build_evaluator_model, env_kwargs, build_env


def init_dir_and_log():
    tf.set_random_seed(FLAGS.seed)
    if FLAGS.use_hdfs:
        base_dir = FLAGS.basedir_hdfs
    else:
        base_dir = FLAGS.basedir_ceph
    base_dir = os.path.join(base_dir, FLAGS.dir)

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
    act_space = int(FLAGS.act_space)
    kwargs["act_space"] = act_space
    """
    get one seg from rollout worker for dtype and shapes

    :param kwargs rollout worker config
    """
    logging.info('get one seg from Evaluator for dtype and shapes')
    ps = AsyncPS.remote()
    small_data_collector = RolloutCollector(
        server_nums=1, ps=ps, policy_evaluator_build_func=build_policy_evaluator,
        **kwargs)
    cache_struct_path = '/tmp/%s.pkl' % FLAGS.dir
    structure = fetch_one_structure(small_data_collector, cache_struct_path=cache_struct_path, is_head=True)
    del small_data_collector

    """
        init data prefetch thread, prepare_input_pipe
    """
    keys = list(structure.keys())
    dtypes = [structure[k].dtype for k in keys]
    shapes = [structure[k].shape for k in keys]
    segBuffer = tf.queue.FIFOQueue(
        capacity=FLAGS.qsize * FLAGS.batch_size,
        dtypes=dtypes,
        shapes=shapes,
        names=keys,
        shared_name="buffer")

    server_nums = FLAGS.nof_evaluator
    nof_server_gpus = FLAGS.nof_server_gpus
    server_nums_refine = server_nums // nof_server_gpus
    data_collector = RolloutCollector(server_nums=server_nums_refine, ps=ps,
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
    demo_buffer = build_demo_buffer(keys, 0.9)
    # //////////////////////
    replay_buffer = PrioritizedReplayBuffer(10000, keys, 0.9)
    weights = tf.placeholder(dtype=tf.float32, shape=[None])

    phs = {key: tf.placeholder(
        dtype=dtype, shape=[None] + list(shape)
    ) for key, dtype, shape in zip(keys, dtypes, shapes)}

    prephs, postphs = dict(), dict()
    for k, v in phs.items():
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

        num_frames_and_train, global_step_and_train, target_op, priority, beta = build_learner(
            pre=predatas, post=postdatas, ws=weights, act_space=act_space, num_frames=num_frames)

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
    sess.run(target_op)
    dequeued_datas, sample_beta = sess.run([dequeued, beta])
    replay_buffer.add_batch(dequeued_datas, FLAGS.batch_size)

    dur_time = 0
    while total_frames < FLAGS.total_environment_frames:
        start = time.time()

        batch_size = np.random.binomial(FLAGS.batch_size - 2, 0.99) + 1
        demo_batch_size = FLAGS.batch_size - batch_size

        datas, is_weights, idxes = replay_buffer.sample(batch_size, sample_beta)
        demo_datas, demo_is_weights, demo_idxes = demo_buffer.sample(demo_batch_size, sample_beta)

        fd = {phs[k]: np.concatenate([datas[k], demo_datas[k]], axis=0) for k in keys}
        fd[weights] = np.concatenate([is_weights, demo_is_weights], axis=0)
        fd[dur_time_tensor] = dur_time

        total_frames, gs, summary, _, p, sample_beta = sess.run(
            [num_frames_and_train, global_step_and_train,
             summary_ops, stage_op, priority, beta], feed_dict=fd)

        replay_buffer.update_priorities(idxes, p[:batch_size])
        demo_buffer.update_priorities(demo_idxes, p[batch_size:])

        if gs % 4 == 0:
            dequeued_datas = sess.run(dequeued)
            replay_buffer.add_batch(dequeued_datas, FLAGS.batch_size)

        if gs % FLAGS.target_update == 0:
            sess.run(target_op)

        if gs % 25 == 0:
            ws = Model.get_ws(sess)
            with open("/opt/tiger/test_ppo/ws.pkl", "wb") as f:
                pickle.dump(ws, f)
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


def main(_):
    if FLAGS.mode == "train":
        train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
