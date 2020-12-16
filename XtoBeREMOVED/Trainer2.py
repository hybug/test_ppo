# coding: utf-8

import tensorflow as tf
import os
import logging
import time
import glob
from queue import Queue
from threading import Thread
import pyarrow as pa

from utils import get_shape
from utils import unpack

from XtoBeREMOVED.PPOcC import PPOcCritic

logging.getLogger('tensorflow').setLevel(logging.ERROR)

nest = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")

flags.DEFINE_string(
    "basedir", "/mnt/cephfs_new_wj/arnold/labcv/xiaochangnan"
               "/PPOcGAE_SuperMarioBros-v0/1", "base dir")
flags.DEFINE_string(
    "datadir", "/mnt/mytmpfs", "data dir")
flags.DEFINE_string(
    "scriptdir", "/opt/tiger/test_ppo", "script dir")

flags.DEFINE_bool("use_stage", True, "whether to use tf.contrib.staging")

flags.DEFINE_integer("num_workers", 4, "number of workers")
flags.DEFINE_integer("worker_parallel", 4, "parallel worker when serve")
flags.DEFINE_integer("max_steps", 3200, "max rollout steps")
flags.DEFINE_integer("seqlen", 64, "seqlen of each rollout segment")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("total_environment_frames", 500000000,
                     "total num of frames for train")

flags.DEFINE_float("init_lr", 5e-3, "initial learning rate")
flags.DEFINE_float("warmup_steps", 4000, "steps for warmup")

flags.DEFINE_integer("frames", 1, "stack of frames for each state")
flags.DEFINE_integer("image_size", 84, "input image size")
flags.DEFINE_float("vf_clip", 1.0, "clip of value function")
flags.DEFINE_float("ppo_clip", 0.2, "clip of ppo loss")
flags.DEFINE_float("gamma", 0.99, "discount rate")
flags.DEFINE_float("vf_coef", 0.1, "weight of value fn loss")
flags.DEFINE_float("ent_coef", 0.0001, "weight of entropy loss")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")

flags.DEFINE_integer("seed", 12358, "random seed")


class Model(object):
    def __init__(self,
                 act_space,
                 vf_clip,
                 lstm,
                 scope="agent",
                 **kwargs):
        self.act_space = act_space
        self.scope = scope

        self.s_t = kwargs.get("s")
        self.previous_actions = kwargs.get("prev_a")
        self.a_t = kwargs.get("a")
        self.old_act_logits = kwargs.get("a_logits")
        self.s_t1 = kwargs.get("s1")

        self.ret = kwargs.get("ret")
        self.advantage = kwargs.get("adv")
        self.target_value = kwargs.get("v_tar")
        self.old_current_value = kwargs.get("v_cur")

        self.c_in = kwargs.get("c_in")
        self.h_in = kwargs.get("h_in")

        self.slots = kwargs.get("slots")

        prev_a = tf.one_hot(
            self.previous_actions, depth=act_space + 1, dtype=tf.float32)

        s_current_feature, self.c_out, self.h_out = self.feature_net(
            self.s_t, lstm, prev_a, scope + "_current_feature")

        self.current_act_logits = self.a_net(
            s_current_feature, scope + "_acurrent")
        # self.current_act = tf.squeeze(
        #     self.categorical(self.current_act_logits, 1), axis=-1)
        self.current_act_probs = tf.nn.softmax(self.current_act_logits, axis=-1)

        # self.old_act = tf.squeeze(
        #     self.categorical(self.old_act_logits, 1), axis=-1)
        self.old_act_probs = tf.nn.softmax(self.old_act_logits, axis=-1)

        self.current_value = self.v_net(
            s_current_feature,
            scope + "_ccurrent")

        self.clip_current_value = self.old_current_value + tf.clip_by_value(
            self.current_value - self.old_current_value, -vf_clip, vf_clip)
        self.current_values = [self.current_value, self.clip_current_value]

    @staticmethod
    def categorical(tensor, num):
        shape = get_shape(tensor)
        if len(shape) == 2:
            return tf.random.categorical(tensor, num)
        elif len(shape) == 3:
            new = tf.reshape(tensor, [-1, shape[-1]])
            sample = tf.random.categorical(new, num)
            return tf.reshape(sample, [shape[0], shape[1], num])
        else:
            raise ValueError(tensor.name + "should have dim 2 or 3")

    def get_return(self):
        return self.ret

    def get_advantage(self):
        return self.advantage

    def get_current_act(self):
        return self.current_act

    def get_current_act_probs(self):
        return self.current_act_probs

    def get_current_act_logits(self):
        return self.current_act_logits

    def get_old_act(self):
        return self.old_act

    def get_old_act_probs(self):
        return self.old_act_probs

    def get_old_act_logits(self):
        return self.old_act_logits

    def get_target_value(self):
        return self.target_value

    def get_current_value(self):
        return self.current_values

    def get_critic_target_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_ctarget"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_target_feature")

    def get_critic_current_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_ccurrent"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_current_feature")

    def get_actor_current_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_acurrent"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_current_feature")

    def get_actor_old_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_aold"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_target_feature")

    def get_variables(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_acurrent"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_ccurrent"
        ) + tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.scope + "_current_feature")

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

    def feature_net(self, image, lstm, prev_a, scope="feature"):
        shape = get_shape(image)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            image = tf.reshape(image, [-1] + shape[-3:])
            filter = [16, 32, 32]
            kernel = [(3, 3), (3, 3), (3, 3)]
            stride = [(1, 2), (2, 2), (1, 2)]
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
            feature = tf.reshape(
                image, [shape[0], shape[1], new_shape[1] * new_shape[2] * new_shape[3]])

            feature = tf.layers.dense(feature, 256, tf.nn.relu, name="feature")
            feature = tf.concat([feature, prev_a], axis=-1)
            c_out, h_out = self.c_in, self.h_in

            feature, c_out, h_out = lstm(
                feature, initial_state=[self.c_in, self.h_in])

        return feature, c_out, h_out

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


def build_learner(pre, post, act_space, num_frames, lstm):
    global_step = tf.train.get_or_create_global_step()
    init_lr = FLAGS.init_lr
    warmup_steps = FLAGS.warmup_steps

    global_step_float = tf.cast(global_step, tf.float32)

    lr = init_lr * warmup_steps ** 0.5 * tf.minimum(
        global_step_float * warmup_steps ** -1.5,
        global_step_float ** -0.5)
    is_warmup = tf.cast(global_step_float < warmup_steps, tf.float32)
    lr = is_warmup * lr + (1.0 - is_warmup) * tf.maximum(lr, FLAGS.init_lr / 16.)
    optimizer = tf.train.AdamOptimizer(lr)

    pre_model = Model(act_space, FLAGS.vf_clip, lstm, "agent", **pre)

    post["c_in"] = tf.stop_gradient(pre_model.c_out)
    post["h_in"] = tf.stop_gradient(pre_model.h_out)

    post_model = Model(act_space, FLAGS.vf_clip, lstm, "agent", **post)

    ppo = PPOcCritic(
        sess=None,
        model=post_model,
        optimizer=optimizer,
        epsilon=FLAGS.ppo_clip,
        c0=FLAGS.vf_coef,
        gamma=FLAGS.gamma,
        entropy=FLAGS.ent_coef,
        actor_freeze_update_per_steps=None,
        actor_freeze_update_step_size=1.0,
        use_freeze=False,
        critic_freeze_update_per_steps=None,
        critic_freeze_update_step_size=1.0,
        trace_decay_rate=None,
        icm=None,
        popart_step_size=None,
        grad_norm_clip=FLAGS.grad_clip,
        scope="ppo")

    new_frames = tf.reduce_sum(post_model.slots)

    with tf.control_dependencies([ppo.ac_current_train_op]):
        num_frames_and_train = tf.assign_add(
            num_frames, new_frames)
        global_step_and_train = tf.assign_add(
            global_step, 1)

    # tf.summary.scalar("learning_rate", lr)
    # tf.summary.scalar("total_loss", ppo.ac_loss)
    # tf.summary.scalar("ppo_loss", ppo.a_loss)
    # tf.summary.scalar("vf_loss", ppo.c_loss)

    return num_frames_and_train, global_step_and_train


class QueueReader(Thread):
    def __init__(self,
                 sess,
                 global_queue,
                 pattern):
        Thread.__init__(self)
        self.daemon = True

        self.sess = sess
        self.global_queue = global_queue
        self.pattern = pattern
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
        self.global_queue.put(seg)

    def next(self):
        while True:
            names = glob.glob(self.pattern)
            if not names:
                time.sleep(1)
                continue
            names.sort(reverse=True)
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


def read_and_remove(name):
    seg = QueueReader.read(name)
    if os.path.exists(name):
        try:
            os.remove(name)
        except FileNotFoundError:
            pass
    if os.path.exists(name[:-9] + ".log"):
        try:
            os.remove(name[:-9] + ".log")
        except FileNotFoundError:
            pass
    if seg is not None:
        return seg[0]
    return None


def train(act_space):
    BASE_DIR = FLAGS.basedir
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)
    CKPT_DIR = os.path.join(BASE_DIR, "ckpt")
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)
    DATA_DIR = FLAGS.datadir
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    SCRIPT_DIR = FLAGS.scriptdir

    os.system("python3 %s/Server.py "
              "-SCRIPT_DIR %s "
              "-BASE_DIR %s "
              "-CKPT_DIR %s "
              "-DATA_DIR %s "
              "-frames %d "
              "-workers %d "
              "-worker_parallel %d "
              "-max_steps %d "
              "-seqlen %d "
              "-vf_clip %.2f "
              "&" % (
                  SCRIPT_DIR,
                  SCRIPT_DIR,
                  BASE_DIR,
                  CKPT_DIR,
                  DATA_DIR,
                  FLAGS.frames,
                  FLAGS.num_workers,
                  FLAGS.worker_parallel,
                  FLAGS.max_steps,
                  FLAGS.seqlen,
                  FLAGS.vf_clip))

    logging.basicConfig(filename=os.path.join(
        BASE_DIR, "Trainerlog"), level="INFO")

    tf.set_random_seed(FLAGS.seed)

    num_frames = tf.get_variable(
        'num_environment_frames',
        initializer=tf.zeros_initializer(),
        shape=[],
        dtype=tf.int32,
        trainable=False)
    global_step = tf.train.get_or_create_global_step()

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    phs = dict()
    phs["s"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, FLAGS.frames], name="s")
    phs["prev_a"] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="prev_a")
    phs["a"] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="a")
    phs["a_logits"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 7], name="a_logits")
    phs["s1"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, FLAGS.frames], name="s1")
    phs["ret"] = tf.placeholder(dtype=tf.float32, shape=[None, None], name="ret")
    phs["adv"] = tf.placeholder(dtype=tf.float32, shape=[None, None], name="adv")
    phs["v_cur"] = tf.placeholder(dtype=tf.float32, shape=[None, None], name="v_cur")
    phs["v_tar"] = tf.placeholder(dtype=tf.float32, shape=[None, None], name="v_tar")
    phs["slots"] = tf.placeholder(dtype=tf.int32, shape=[None, None], name="slots")

    prephs = {k: tf.split(v, 2, axis=1)[0] for k, v in phs.items()}
    postphs = {k: tf.split(v, 2, axis=1)[1] for k, v in phs.items()}

    phs["c_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 256], name="c_in")
    phs["h_in"] = tf.placeholder(dtype=tf.float32, shape=[None, 256], name="h_in")
    prephs["c_in"] = phs["c_in"]
    prephs["h_in"] = phs["h_in"]

    prekeys = list(prephs.keys())
    postkeys = list(postphs.keys())
    keys = list(phs.keys())

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

    segBuffer = Queue(maxsize=2 * FLAGS.batch_size)
    readers = []
    patterns = [os.path.join(
        DATA_DIR, "*_%s_*.seg" % ((4 - len(str(i))) * "0" + str(i))
    ) for i in range(FLAGS.num_workers)]
    for pattern in patterns:
        reader = QueueReader(
            sess=sess,
            global_queue=segBuffer,
            pattern=pattern)
        reader.start()
        readers.append(reader)

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

        lstm = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")
        num_frames_and_train, global_step_and_train = build_learner(
            pre=predatas, post=postdatas, act_space=act_space,
            num_frames=num_frames, lstm=lstm)

    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=6)

    ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    saver.save(sess, os.path.join(CKPT_DIR, "PPOcGAE"), global_step=global_step)

    total_frames = 0

    batch = []
    for i in range(FLAGS.batch_size):
        batch.append(segBuffer.get())

    fd = dict()
    for key in keys:
        fd[phs[key]] = [seg[key] for seg in batch]
    sess.run(stage_op, feed_dict=fd)

    while total_frames < FLAGS.total_environment_frames:
        start = time.time()

        batch = []
        for i in range(FLAGS.batch_size):
            batch.append(segBuffer.get())

        fd = dict()
        for key in keys:
            fd[phs[key]] = [seg[key] for seg in batch]

        total_frames, gs, _ = sess.run(
            [num_frames_and_train,
             global_step_and_train,
             stage_op],
            feed_dict=fd)

        if gs % 25 == 0:
            saver.save(sess, os.path.join(CKPT_DIR, "PPOcGAE"), global_step=global_step)

        if gs % 1 == 0:
            logging.info(
                "Global Step %d, Total Frames %d,  Time Consume %.2f" % (
                    gs, total_frames, time.time() - start))


if __name__ == '__main__':
    train(7)
    pass
