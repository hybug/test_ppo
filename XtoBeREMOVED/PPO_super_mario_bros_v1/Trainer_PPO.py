# coding: utf-8

import sys

sys.path.append("/opt/tiger/test_ppo")

import tensorflow as tf
import numpy as np
import os
import logging
import time
import glob
from threading import Thread
import pyarrow as pa

from utils import get_shape
from utils import unpack
from module import entropy_from_logits as entropy
from module import RMCRNN, TmpHierRMCRNN
from module import KL_from_gaussians
from module import icm
from module import coex
from module import mse
from infer import categorical
from train_ops import miniOp
from algorithm import dPPOcC

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.getLogger('tensorflow').setLevel(logging.ERROR)

nest = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")
flags.DEFINE_integer("act_space", 12, "act space")

flags.DEFINE_integer(
    "cluster", 2, "base dir")
flags.DEFINE_string("dir", "0", "dir number")
flags.DEFINE_string(
    "datadir", "/mnt/mytmpfs", "data dir")
flags.DEFINE_string(
    "scriptdir", "/opt/tiger/test_ppo/examples/PPO_super_mario_bros_v1", "script dir")

flags.DEFINE_bool("use_stage", True, "whether to use tf.contrib.staging")
flags.DEFINE_bool("use_rmc", True, "whether to use rmcrnn instead of lstm")
flags.DEFINE_bool("use_hrmc", True, "whether to use tmp hierarchy rmcrnn")
flags.DEFINE_bool("use_icm", False, "whether to use icm during training")
flags.DEFINE_bool("use_coex", False, "whether to use coex adm during training")
flags.DEFINE_bool("use_reward_prediction", True, "whether to use reward prediction")
flags.DEFINE_bool("use_pixel_control", True, "whether to use pixel control")

flags.DEFINE_integer("num_servers", 8, "number of servers")
flags.DEFINE_integer("num_workers", 4, "number of workers")
flags.DEFINE_integer("worker_parallel", 64, "parallel workers")
flags.DEFINE_integer("max_steps", 3200, "max rollout steps")
flags.DEFINE_integer("seqlen", 32, "seqlen of each training segment")
flags.DEFINE_integer("burn_in", 32, "seqlen of each burn-in segment")
flags.DEFINE_integer("batch_size", 128, "batch size")
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
flags.DEFINE_float("pq_kl_coef", 0.1, "weight of kl between posterior and prior")
flags.DEFINE_float("p_kl_coef", 0.01, "weight of kl between prior and normal gaussian")
flags.DEFINE_bool("zero_init", False, "whether to zero init initial state")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")

flags.DEFINE_integer("seed", 12358, "random seed")


class Model(object):
    def __init__(self,
                 act_space,
                 rnn,
                 use_rmc,
                 use_hrmc,
                 use_reward_prediction,
                 after_rnn,
                 use_pixel_control,
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

            if use_pixel_control:
                self.pixel_control = self.reconstruct_net(self.feature)

    def get_current_act(self):
        return self.current_act

    def get_current_act_logits(self):
        return self.current_act_logits

    def v_net(self, feature, scope):
        net = feature
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # net = tf.layers.dense(
            #     net,
            #     get_shape(feature)[-1],
            #     activation=tf.nn.relu,
            #     name="dense")
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
            # net = tf.layers.dense(
            #     net,
            #     get_shape(feature)[-1],
            #     activation=tf.nn.relu,
            #     name="dense")
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


def build_learner(pre, post, act_space, num_frames):
    global_step = tf.train.get_or_create_global_step()
    init_lr = FLAGS.init_lr
    decay = FLAGS.lr_decay
    warmup_steps = FLAGS.warmup_steps
    use_rmc = FLAGS.use_rmc
    use_hrmc = FLAGS.use_hrmc
    use_icm = FLAGS.use_icm
    use_coex = FLAGS.use_coex
    use_reward_prediction = FLAGS.use_reward_prediction
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

    ent_coef = tf.train.polynomial_decay(
        FLAGS.ent_coef, global_step,
        FLAGS.total_environment_frames * 2 // (
                FLAGS.batch_size * FLAGS.seqlen),
        FLAGS.ent_coef / 10.)

    if FLAGS.zero_init:
        pre["state_in"] = tf.zeros_like(pre["state_in"])

    if use_hrmc:
        rnn = TmpHierRMCRNN(
            4, 64, 4, 4, 4, return_sequences=True, return_state=True, name="hrmcrnn")
    elif use_rmc:
        rnn = RMCRNN(
            64, 4, 4, return_sequences=True, return_state=True, name="rmcrnn")
    else:
        rnn = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")
    pre_model = Model(
        act_space, rnn, use_rmc, use_hrmc,
        use_reward_prediction, use_pixel_control, "agent", **pre)

    post["state_in"] = tf.stop_gradient(pre_model.state_out)

    post_model = Model(
        act_space, rnn, use_rmc, use_hrmc,
        use_reward_prediction, use_pixel_control, "agent", **post)

    tf.summary.scalar("adv_mean", post_model.adv_mean)
    tf.summary.scalar("adv_std", post_model.adv_std)

    losses = dPPOcC(
        act=post_model.a_t,
        policy_logits=post_model.current_act_logits,
        old_policy_logits=post_model.old_act_logits,
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
    if use_hrmc:
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
        rec_loss = tf.reduce_mean(
            mse(post_model.pixel_control, post_model.s_t
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


class QueueReader(Thread):
    def __init__(self,
                 sess,
                 global_queue,
                 pattern,
                 keys,
                 dtypes,
                 shapes):
        Thread.__init__(self)
        self.daemon = True

        self.sess = sess
        self.global_queue = global_queue
        self.pattern = pattern

        self.keys = keys
        self.placeholders = [
            tf.placeholder(
                dtype=dtype, shape=shape
            ) for dtype, shape in zip(dtypes, shapes)]
        self.enqueue_op = self.global_queue.enqueue(
            dict(zip(keys, self.placeholders)))
        self.generator = self.next()

        self.count = 0
        self.retime = 0
        self.untime = 0

    @staticmethod
    def read(name):
        try:
            start_time = time.time()
            with pa.OSFile(name) as f:
                s = f.read_buffer()
            readtime = time.time() - start_time
            start_time = time.time()
            s = unpack(s)
            untime = time.time() - start_time
            return s, readtime, untime
        except Exception as e:
            logging.warning(e)
            return None

    def enqueue(self):
        seg = next(self.generator)
        fd = {self.placeholders[i]: seg[key] for i, key in enumerate(self.keys)}
        self.sess.run(self.enqueue_op, fd)

    def next(self):
        while True:
            names = glob.glob(self.pattern)
            if not names:
                time.sleep(1)
                continue
            np.random.shuffle(names)
            while names:
                name = names.pop()
                seg = self.read(name)
                if seg is not None:
                    if os.path.exists(name):
                        os.remove(name)
                    if os.path.exists(name[:-9] + ".log"):
                        os.remove(name[:-9] + ".log")
                    seg, retime, untime = seg
                    # while self.global_queue.qsize() >= self.max_size:
                    #     time.sleep(1)
                    # self.global_queue.enqueue(seg)
                    self.count += 1
                    self.retime += retime
                    self.untime += untime
                    if self.count % 100 == 0:
                        # logging.info(
                        #     "Read time %.2f, Unpack time %.2f"
                        #     % (self.retime, self.untime))
                        self.count = 0
                        self.retime = 0
                        self.untime = 0
                    yield seg

    def run(self):
        while True:
            self.enqueue()


def train():
    act_space = FLAGS.act_space
    if FLAGS.cluster == 2 or 4:
        basedir = "/mnt/cephfs_new_wj/arnold/labcv/" \
                  "xiaochangnan/PPOcGAE_SuperMarioBros-v0"
    else:
        basedir = "/mnt/mesos/xiaochangnan"
    BASE_DIR = os.path.join(basedir, FLAGS.dir)
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    CKPT_DIR = os.path.join(BASE_DIR, "ckpt")
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)
    DATA_DIR = FLAGS.datadir
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    SCRIPT_DIR = FLAGS.scriptdir

    for i in range(FLAGS.num_servers):
        os.system("python3 %s/Server_PPO.py "
                  "-server_id %d "
                  "-SCRIPT_DIR %s "
                  "-BASE_DIR %s "
                  "-CKPT_DIR %s "
                  "-DATA_DIR %s "
                  "-frames %d "
                  "-workers %d "
                  "-worker_parallel %d "
                  "-max_steps %d "
                  "-seqlen %d "
                  "-burn_in %d "
                  "-act_space %d "
                  "-use_rmc %d "
                  "-use_hrmc %d "
                  "-use_reward_prediction %d "
                  "-use_pixel_control %d "
                  "&" % (
                      SCRIPT_DIR,
                      i,
                      SCRIPT_DIR,
                      BASE_DIR,
                      CKPT_DIR,
                      DATA_DIR,
                      FLAGS.frames,
                      FLAGS.num_workers,
                      FLAGS.worker_parallel,
                      FLAGS.max_steps,
                      FLAGS.seqlen,
                      FLAGS.burn_in,
                      FLAGS.act_space,
                      FLAGS.use_rmc,
                      FLAGS.use_hrmc,
                      FLAGS.use_reward_prediction,
                      FLAGS.use_pixel_control))

    logging.basicConfig(filename=os.path.join(
        BASE_DIR, "Trainerlog"), level="INFO")

    tf.set_random_seed(FLAGS.seed)

    num_frames = tf.get_variable(
        'num_environment_frames',
        initializer=tf.zeros_initializer(),
        shape=[],
        dtype=tf.int32,
        trainable=False)
    tf.summary.scalar("frames", num_frames)
    global_step = tf.train.get_or_create_global_step()

    config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.94))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    structure = None
    while True:
        names = glob.glob(os.path.join(DATA_DIR, "*.seg"))
        while names:
            name = names.pop()
            seg = QueueReader.read(name)
            if seg is not None:
                structure = seg[0]
                flatten_structure = nest.flatten(structure)
                break
        if structure is not None:
            break
        logging.warning("NO DATA, SLEEP %d seconds" % 60)
        time.sleep(60)

    keys = list(structure.keys())
    dtypes = [structure[k].dtype for k in keys]
    shapes = [structure[k].shape for k in keys]
    segBuffer = tf.queue.FIFOQueue(
        capacity=2 * FLAGS.batch_size,
        dtypes=dtypes,
        shapes=shapes,
        names=keys,
        shared_name="buffer")

    readers = []
    patterns = [os.path.join(
        DATA_DIR, "*_%s_*_*.seg" % ((4 - len(str(i))) * "0" + str(i))
    ) for i in range(FLAGS.num_workers)]
    for pattern in patterns:
        reader = QueueReader(
            sess=sess,
            global_queue=segBuffer,
            pattern=pattern,
            keys=keys,
            dtypes=dtypes,
            shapes=shapes)
        reader.start()
        readers.append(reader)

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

        num_frames_and_train, global_step_and_train = build_learner(
            pre=predatas, post=postdatas, act_space=act_space,
            num_frames=num_frames)

    summary_ops = tf.summary.merge_all()
    if FLAGS.cluster == 2 or 4:
        summary_writer = tf.summary.FileWriter(os.path.join(
            BASE_DIR, "summary"), sess.graph)
    else:
        summary_writer = tf.summary.FileWriter(os.path.join(
            os.getenv("ARNOLD_OUTPUT"), "ppo%s" % FLAGS.dir), sess.graph)

    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=6)

    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    saver.save(sess, os.path.join(CKPT_DIR, "PPOcGAE"), global_step=global_step)

    total_frames = 0
    sess.run(stage_op)

    while total_frames < FLAGS.total_environment_frames:
        start = time.time()

        total_frames, gs, summary, _ = sess.run(
            [num_frames_and_train, global_step_and_train, summary_ops, stage_op])
        summary_writer.add_summary(summary, global_step=gs)

        if gs % 25 == 0:
            saver.save(sess, os.path.join(CKPT_DIR, "PPOcGAE"), global_step=global_step)

        if gs % 1 == 0:
            msg = "Global Step %d, Total Frames %d,  Time Consume %.2f" % (
                gs, total_frames, time.time() - start)
            logging.info(msg)


def main(_):
    if FLAGS.mode == "train":
        train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
