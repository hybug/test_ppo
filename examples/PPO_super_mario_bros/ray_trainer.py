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
import tensorflow as tf

from utils import get_shape
from module import RMCRNN
from train_ops import miniOp
from module import entropy_from_logits as entropy
from module import TmpHierRMCRNN
from module import TmpHierRNN
from module import icm
from module import coex
from module import mse
from module import KL_from_gaussians
from algorithm import dPPOcC

logging.getLogger('tensorflow').setLevel(logging.ERROR)

NEST = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")
flags.DEFINE_integer("act_space", 12, "act space")

flags.DEFINE_string(
    "basedir_ceph", "/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
                    "/PPOcGAE_SuperMarioBros-v0", "base dir for ceph")
flags.DEFINE_string(
    "basedir_hdfs",
    'hdfs://haruna/home/byte_arnold_lq_mlsys/user/xiaochangnan/test_ppo',
    "base dir for hdfs")
flags.DEFINE_string("dir", "0", "dir number")
flags.DEFINE_string(
    "scriptdir",
    "/opt/tiger/test_ppo/examples/PPO_super_mario_bros",
    "script dir")

flags.DEFINE_bool("use_stage", True, "whether to use tf.contrib.staging")

flags.DEFINE_integer("use_rmc", 0, "whether to use rmcrnn instead of lstm")
flags.DEFINE_integer("use_hrmc", 1, "whether to use tmp hierarchy rmcrnn")
flags.DEFINE_integer("use_hrnn", 0, "whether to use tmp hierarchy rnn (lstm+rmc)")
flags.DEFINE_bool("use_icm", False, "whether to use icm during training")
flags.DEFINE_bool("use_coex", False, "whether to use coex adm during training")
flags.DEFINE_bool("use_reward_prediction", True, "whether to use reward prediction")
flags.DEFINE_integer("after_rnn", 1, "whether to use reward prediction after rnn")
flags.DEFINE_integer("use_pixel_control", 1, "whether to use pixel control")
flags.DEFINE_integer("use_pixel_reconstruction", 0, "whether to use pixel reconstruction")

flags.DEFINE_float("pq_kl_coef", 0.1, "weight of kl between posterior and prior")
flags.DEFINE_float("p_kl_coef", 0.01, "weight of kl between prior and normal gaussian")

flags.DEFINE_bool("use_hdfs", True, "whether to use hdfs")

flags.DEFINE_integer("parallel", 64, "parallel envs")
flags.DEFINE_integer("max_steps", 3200, "max rollout steps")
flags.DEFINE_integer("seqlen", 32, "seqlen of each training segment")
flags.DEFINE_integer("burn_in", 32, "seqlen of each burn-in segment")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("total_environment_frames", 1000000000,
                     "total num of frames for train")

flags.DEFINE_float("init_lr", 1e-3, "initial learning rate")
flags.DEFINE_float("lr_decay", 1.0, "whether to decay learning rate")
flags.DEFINE_float("warmup_steps", 4000, "steps for warmup")

flags.DEFINE_integer("frames", 1, "stack of frames for each state")
flags.DEFINE_integer("image_size", 84, "input image size")
flags.DEFINE_float("vf_clip", 1.0, "clip of value function")
flags.DEFINE_float("ppo_clip", 0.2, "clip of ppo loss")
flags.DEFINE_float("gamma", 0.99, "discount rate")
flags.DEFINE_float("pi_coef", 10.0, "weight of policy fn loss")
flags.DEFINE_float("vf_coef", 1.0, "weight of value fn loss")
flags.DEFINE_float("ent_coef", 1.0, "weight of entropy loss")
flags.DEFINE_bool("zero_init", False, "whether to zero init initial state")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")

flags.DEFINE_integer("seed", 12358, "random seed")

# param for ray evaluator
flags.DEFINE_float("timeout", 0.1, "get operation timeout")
flags.DEFINE_integer("num_returns", 32, "nof data of wait operation")
flags.DEFINE_integer('cpu_per_actor', 2, 'nof cpu required for infserver, -1 for not require')
flags.DEFINE_integer('load_ckpt_period', 10, 'for how many step to load ckpt in inf server')
flags.DEFINE_integer('qsize', 8, 'for how many qsize * batchsize in main procress')
flags.DEFINE_integer('nof_server_gpus', 1, 'nof_gpus for training')
flags.DEFINE_integer('nof_evaluator', 1, 'nof_gpus for training')

from examples.PPO_super_mario_bros.policy_graph import warp_Model

Model = warp_Model()


def build_learner(pre, post, act_space, num_frames):
    global_step = tf.train.get_or_create_global_step()
    init_lr = FLAGS.init_lr
    decay = FLAGS.lr_decay
    warmup_steps = FLAGS.warmup_steps
    use_rmc = FLAGS.use_rmc
    use_hrmc = FLAGS.use_hrmc
    use_hrnn = FLAGS.use_hrnn
    use_icm = FLAGS.use_icm
    use_coex = FLAGS.use_coex
    use_reward_prediction = FLAGS.use_reward_prediction
    after_rnn = FLAGS.after_rnn
    use_pixel_control = FLAGS.use_pixel_control
    use_pixel_reconstruction = FLAGS.use_pixel_reconstruction
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

    ent_coef = tf.train.polynomial_decay(
        FLAGS.ent_coef, global_step,
        FLAGS.total_environment_frames // (
                FLAGS.batch_size * FLAGS.seqlen),
        FLAGS.ent_coef / 10.)

    if FLAGS.zero_init:
        pre["state_in"] = tf.zeros_like(pre["state_in"])

    if use_hrnn:
        rnn = TmpHierRNN(4, 64, 4, 2, 8, 'lstm', 'rmc',
                         return_sequences=True, return_state=True, name="hrnn")
    elif use_hrmc:
        rnn = TmpHierRMCRNN(
            4, 64, 4, 4, return_sequences=True, return_state=True, name="hrmcrnn")
    elif use_rmc:
        rnn = RMCRNN(
            64, 4, 4, return_sequences=True, return_state=True, name="rmcrnn")
    else:
        rnn = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")
    pre_model = Model(
        act_space, rnn, use_rmc, use_hrmc or use_hrnn,
        use_reward_prediction, after_rnn, use_pixel_control,
        use_pixel_reconstruction, "agent", **pre)

    post["state_in"] = tf.stop_gradient(pre_model.state_out)

    post_model = Model(
        act_space, rnn, use_rmc, use_hrmc or use_hrnn,
        use_reward_prediction, after_rnn, use_pixel_control,
        use_pixel_reconstruction, "agent", **post)

    tf.summary.scalar("adv_mean", post_model.adv_mean)
    tf.summary.scalar("adv_std", post_model.adv_std)

    losses = dPPOcC(
        act=post_model.a_t,
        policy_logits=post_model.current_act_logits,
        behavior_logits=post_model.behavior_logits,
        advantage=post_model.advantage,
        policy_clip=FLAGS.ppo_clip,
        vf=post_model.current_value,
        vf_target=post_model.ret,
        value_clip=FLAGS.vf_clip,
        old_vf=post_model.old_current_value)

    entropy_loss = tf.reduce_mean(
        entropy(post_model.current_act_logits) * post_model.slots)

    p_loss = tf.reduce_mean(losses.p_loss * post_model.slots)
    v_loss = tf.reduce_mean(losses.v_loss * post_model.slots)

    add_loss = 0.0
    if use_icm:
        icmloss = icm(
            post_model.cnn_feature[:, :-1, :],
            post_model.cnn_feature[:, 1:, :],
            post_model.a_t[:, :-1],
            act_space)
        add_loss += 0.2 * tf.reduce_mean(
            icmloss.f_loss * post_model.slots[:, :-1]
        ) + 0.8 * tf.reduce_mean(
            icmloss.i_loss * post_model.slots[:, :-1])
    if use_coex:
        coexloss = coex(
            post_model.image_feature[:, :-1, :, :, :],
            post_model.image_feature[:, 1:, :, :, :],
            post_model.a_t[:, :-1],
            act_space)
        add_loss += tf.reduce_mean(
            coexloss * post_model.slots[:, :-1])
    if use_hrmc or use_hrnn:
        pq_kl_loss = KL_from_gaussians(
            post_model.q_mus, post_model.q_sigmas,
            post_model.p_mus, post_model.p_sigmas)
        pq_kl_loss = tf.reduce_mean(pq_kl_loss * post_model.slots)
        tf.summary.scalar("kl_div", pq_kl_loss)
        add_loss += pq_kl_coef * pq_kl_loss

        p_kl_loss = KL_from_gaussians(
            post_model.p_mus, post_model.p_sigmas,
            tf.zeros_like(post_model.p_mus), 0.01 * tf.ones_like(post_model.p_sigmas))
        p_kl_loss = tf.reduce_mean(p_kl_loss * post_model.slots)
        tf.summary.scalar("kl_div_prior", p_kl_loss)
        add_loss += p_kl_coef * p_kl_loss
    if use_reward_prediction:
        r_loss = tf.reduce_mean(
            mse(post_model.reward_prediction, post_model.r_t) * post_model.slots)
        tf.summary.scalar("r_loss", r_loss)
        add_loss += r_loss
    if use_pixel_control:
        change_of_cells = tf.reduce_mean(
            post_model.s_t[:, 1:, :, :, :] - post_model.s_t[:, :-1, :, :, :], axis=-1)
        s_shape = get_shape(change_of_cells)
        s_H, s_W = s_shape[2:]
        ctr_H, ctr_W = get_shape(post_model.pixel_control)[2: 4]
        change_of_cells = tf.reduce_mean(
            tf.reshape(
                change_of_cells,
                shape=s_shape[:2] + [ctr_H, s_H // ctr_H, ctr_W, s_W // ctr_W]),
            axis=(3, 5))

        ctr = tf.reduce_sum(
            tf.transpose(
                post_model.pixel_control, perm=(0, 1, 4, 2, 3)
            ) * tf.one_hot(
                post_model.a_t,
                depth=post_model.act_space,
                dtype=tf.float32
            )[:, :, :, None, None],
            axis=2)[:, :-1, :, :]
        ctr_loss = tf.reduce_mean(
            mse(ctr, change_of_cells)
        )
        tf.summary.scalar("pixel_control_loss", ctr_loss)
        add_loss += ctr_loss
    if use_pixel_reconstruction:
        rec_loss = tf.reduce_mean(
            mse(post_model.pixel_reconstruction, post_model.s_t
                ) * post_model.slots[:, :, None, None, None])
        tf.summary.scalar("rec_loss", rec_loss)
        add_loss += rec_loss

    loss = (FLAGS.pi_coef * p_loss
            + FLAGS.vf_coef * v_loss
            - ent_coef * entropy_loss
            + add_loss)

    train_op = miniOp(optimizer, loss, FLAGS.grad_clip)

    new_frames = tf.reduce_sum(post["slots"])

    with tf.control_dependencies([train_op]):
        num_frames_and_train = tf.assign_add(
            num_frames, new_frames)
        global_step_and_train = tf.assign_add(
            global_step, 1)

    tf.summary.scalar("learning_rate", lr)
    tf.summary.scalar("ent_coef", ent_coef)
    tf.summary.scalar("ent_loss", entropy_loss)
    tf.summary.scalar("p_loss", p_loss)
    tf.summary.scalar("v_loss", v_loss)
    tf.summary.scalar("all_loss", loss)

    return num_frames_and_train, global_step_and_train


def build_policy_evaluator(kwargs):
    """
    construct policy_evaluator
    :param kwargs:
    :return: construcation method and params for Evaluator
    """
    from copy import deepcopy

    if kwargs["use_hrnn"]:
        kwargs["state_size"] = 1 + (8 + 2 + 8) * 4 * 64
    elif kwargs["use_hrmc"]:
        kwargs["state_size"] = 1 + (8 + 4 + 4) * 4 * 64
    elif kwargs["use_rmc"]:
        kwargs['state_size'] = 64 * 4 * 4
    else:
        kwargs['state_size'] = 256 * 2

    env_kwargs = deepcopy(kwargs)
    env_kwargs['action_repeats'] = [1]

    model_kwargs = deepcopy(kwargs)
    # pickle func in func
    from examples.PPO_super_mario_bros.env import build_env
    from examples.PPO_super_mario_bros.policy_graph import build_evaluator_model
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
    server_nums_refine = server_nums * 2 // FLAGS.cpu_per_actor
    nof_server_gpus = FLAGS.nof_server_gpus
    server_nums_refine = server_nums_refine // nof_server_gpus
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
    prephs, postphs = dict(), dict()
    for k, v in dequeued.items():
        if k == "state_in":
            prephs[k] = v
        else:
            prephs[k], postphs[k] = tf.split(
                v, [FLAGS.burn_in, FLAGS.seqlen], axis=1)
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

        act_space = FLAGS.act_space
        num_frames_and_train, global_step_and_train = build_learner(
            pre=predatas, post=postdatas, act_space=act_space,
            num_frames=num_frames)

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

    saver.save(sess, os.path.join(ckpt_dir, "PPOcGAE"), global_step=global_step)

    """
        step
    """
    total_frames = 0
    sess.run(stage_op)

    dur_time = 0
    while total_frames < FLAGS.total_environment_frames:
        start = time.time()

        total_frames, gs, summary, _ = sess.run(
            [num_frames_and_train, global_step_and_train, summary_ops, stage_op],
            feed_dict={dur_time_tensor: dur_time})

        if gs % 25 == 0:
            ws = Model.get_ws(sess)
            logging.info('pushing weight to ps')
            ray.get(ps.push.remote(ws))

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
