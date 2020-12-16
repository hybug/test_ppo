# coding: utf-8

import tensorflow as tf
import contextlib
import time
import glob
import os

import py_process
from XtoBeREMOVED.PPOcC import PPOcCritic
from utils import get_shape
from utils import unpack

nest = tf.contrib.framework.nest

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "mode")

flags.DEFINE_string("datadir", None, "data dir")
flags.DEFINE_string("logdir", None, "logdir")

flags.DEFINE_string("job_name", "learner", "job name")
flags.DEFINE_integer("task", 0, "task index")

flags.DEFINE_integer("num_workers", 32, "number of workers")
flags.DEFINE_integer("batch_size", 512, "batch size")
flags.DEFINE_integer("total_environment_frames", 100000000,
                     "total num of frames for train")

flags.DEFINE_float("init_lr", 1e-4, "initial learning rate")
flags.DEFINE_float("warmup_steps", 4000, "steps for warmup")

flags.DEFINE_integer("frames", 1, "stack of frames for each state")
flags.DEFINE_integer("image_size", 84, "input image size")
flags.DEFINE_float("vf_clip", 1.0, "clip of value function")
flags.DEFINE_float("ppo_clip", 0.2, "clip of ppo loss")
flags.DEFINE_float("gamma", 0.99, "discount rate")
flags.DEFINE_float("vf_coef", 1.0, "weight of value fn loss")
flags.DEFINE_float("ent_coef", 0.01, "weight of entropy loss")
flags.DEFINE_float("grad_clip", 40.0, "global grad clip")

flags.DEFINE_integer("seed", 12358, "random seed")


@contextlib.contextmanager
def pin_global_variables(device):
    """Pins global variables to the specified device."""

    def getter(getter, *args, **kwargs):
        var_collections = kwargs.get('collections', None)
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
            with tf.device(device):
                return getter(*args, **kwargs)
        else:
            return getter(*args, **kwargs)

    with tf.variable_scope('', custom_getter=getter) as vs:
        yield vs


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
        self.current_act = tf.squeeze(
            self.categorical(self.current_act_logits, 1), axis=-1)
        self.current_act_probs = tf.nn.softmax(self.current_act_logits, axis=-1)

        self.old_act = tf.squeeze(
            self.categorical(self.old_act_logits, 1), axis=-1)
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
        seq_length = shape[1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            image = tf.reshape(image, [-1] + shape[-3:])
            # image = image / 255.
            # image = tf.image.resize_bilinear(image, (84, 84))
            filter = [16, 32, 64, 128]
            kernel = [(7, 7), (5, 5), (3, 3), (3, 3)]
            stride = [4, 2, 2, 1]
            fn = ["conv", "conv", "conv", "conv"]
            for i in range(4):
                if fn[i] == "conv":
                    image = tf.layers.conv2d(
                        image,
                        filters=filter[i],
                        kernel_size=kernel[i],
                        strides=stride[i],
                        padding="valid",
                        activation=tf.nn.relu,
                        name="%s_%d" % (fn[i], i))
                elif fn[i] == "maxpool":
                    image = tf.layers.max_pooling2d(
                        image,
                        pool_size=kernel[i],
                        strides=stride[i],
                        padding="valid",
                        name="%s_%d" % (fn[i], i))
            new_shape = get_shape(image)
            feature = tf.reshape(
                image, [shape[0], shape[1], new_shape[1] * new_shape[2] * new_shape[3]])

            feature = tf.layers.dense(feature, 128, tf.nn.relu, name="feature")
            feature = tf.concat([feature, prev_a], axis=-1)
            c_out, h_out = self.c_in, self.h_in

            feature, c_out, h_out = lstm(
                feature, initial_state=[self.c_in, self.h_in])

        return feature, c_out, h_out


def build_learner(batch, act_space, num_frames):
    global_step = tf.train.get_or_create_global_step()
    gsop = tf.assign_add(global_step, 1)
    init_lr = FLAGS.init_lr
    warmup_steps = FLAGS.warmup_steps

    lr = init_lr * warmup_steps ** 0.5 * tf.minimum(
        global_step * warmup_steps ** -1.5, global_step ** -0.5)
    optimizer = tf.train.AdamOptimizer(lr)

    lstm = tf.compat.v1.keras.layers.LSTM(
        128, return_sequences=True, return_state=True, name="lstm")

    batch = {k: tf.split(v, 2, axis=1) for k, v in batch.items()}

    pre = {k: v[0] for k, v in batch.items()}
    pre["c_in"] = pre["c_in"][:, 0, :]
    pre["h_in"] = pre["h_in"][:, 0, :]

    pre_model = Model(act_space, FLAGS.vf_clip, lstm, "agent", **pre)

    post = {k: v[1] for k, v in batch.items()}
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

    new_frames = tf.reduce_sum(post["slots"])

    with tf.control_dependencies([ppo.ac_current_train_op, gsop]):
        num_frames_and_train = tf.assign_add(
            num_frames, new_frames)

    tf.summary.scalar("learning_rate", lr)
    tf.summary.scalar("total_loss", ppo.ac_loss)
    tf.summary.scalar("ppo_loss", ppo.a_loss)
    tf.summary.scalar("vf_loss", ppo.c_loss)

    return num_frames_and_train


def build_worker(data_dir, pattern):
    def read(name):
        try:
            with open(name, "r") as f:
                s = f.readline()
            s = unpack(s)
            return s
        except Exception as e:
            tf.logging.warning(e)
            return None

    pattern = os.path.join(data_dir, pattern)
    find = None
    while True:
        while True:
            names = glob.glob(pattern)
            if names:
                break
            time.sleep(1)
        names.sort(reverse=True)
        while names:
            name = names.pop()
            seg = read(name)
            os.system("rm %s" % name)
            if seg is not None:
                find = seg
                break
        if find is not None:
            break
    return find


def train(act_space):
    local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
    shared_job_device = '/job:learner/task:0'
    is_worker_fn = lambda i: FLAGS.job_name == 'worker' and i == FLAGS.task
    is_learner = FLAGS.job_name == 'learner'

    # Placing the variable on CPU, makes it cheaper to send it to all the
    # workers. Continual copying the variables from the GPU is slow.
    global_variable_device = shared_job_device + '/cpu'
    cluster = tf.train.ClusterSpec({
        'worker': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_workers)],
        'learner': ['localhost:8000']
    })
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task)
    filters = [shared_job_device, local_job_device]

    # Only used to find the worker output structure.
    with tf.Graph().as_default():
        structure = build_worker(FLAGS.datadir, "*.seg")
        flattened_structure = nest.flatten(structure)
        dtypes = [t.dtype for t in flattened_structure]
        shapes = [t.shape.as_list() for t in flattened_structure]

    with tf.Graph().as_default(), \
         tf.device(local_job_device + '/cpu'), \
         pin_global_variables(global_variable_device):
        tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

        # Create Queue and Agent on the learner.
        with tf.device(shared_job_device):
            queue = tf.FIFOQueue(
                2 * FLAGS.batch_size, dtypes, shapes, shared_name='buffer')
            model = Model(act_space, FLAGS.frames, FLAGS.vf_clip)

        # Build workers and ops to enqueue their output.
        enqueue_ops = []
        for i in range(FLAGS.num_workers):
            if is_worker_fn(i):
                tf.logging.info('Creating worker %d', i)
                pattern = "*_%s_*.seg" % ((4 - len(str(i))) * "0" + str(i))
                worker_output = build_worker(FLAGS.datadir, pattern)
                with tf.device(shared_job_device):
                    enqueue_ops.append(queue.enqueue(nest.flatten(worker_output)))

        # Build learner.
        if is_learner:
            # Create global step, which is the number of environment frames processed.
            num_frames = tf.get_variable(
                'num_environment_frames',
                initializer=tf.zeros_initializer(),
                shape=[],
                dtype=tf.int64,
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            # Create batch (time major) and recreate structure.
            dequeued = queue.dequeue_many(FLAGS.batch_size)
            dequeued = nest.pack_sequence_as(structure, dequeued)

            with tf.device('/gpu'):
                # Using StagingArea allows us to prepare the next batch and send it to
                # the GPU while we're performing a training step. This adds up to 1 step
                # policy lag.
                flattened_output = nest.flatten(dequeued)
                area = tf.contrib.staging.StagingArea(
                    [t.dtype for t in flattened_output],
                    [t.shape for t in flattened_output])
                stage_op = area.put(flattened_output)

                data_from_workers = nest.pack_sequence_as(structure, area.get())

                # Unroll agent on sequence, create losses and update ops.
                output = build_learner(data_from_workers, act_space, num_frames)

        # Create MonitoredSession (to run the graph, checkpoint and log).
        tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
        config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
        with tf.train.MonitoredTrainingSession(
                server.target,
                is_chief=is_learner,
                checkpoint_dir=FLAGS.logdir,
                save_checkpoint_secs=600,
                save_summaries_secs=30,
                log_step_count_steps=50000,
                config=config,
                hooks=[py_process.PyProcessHook()]) as session:

            if is_learner:
                # Logging.
                # level_returns = {level_name: [] for level_name in level_names}
                # summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

                # Prepare data for first run.
                session.run_step_fn(
                    lambda step_context: step_context.session.run(stage_op))

                # Execute learning and track performance.
                num_env_frames_v = 0
                while num_env_frames_v < FLAGS.total_environment_frames:
                    num_env_frames_v, _ = session.run([output, stage_op])
                    # level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)
                    #
                    # for level_name, episode_return, episode_step in zip(
                    #         level_names_v[done_v],
                    #         infos_v.episode_return[done_v],
                    #         infos_v.episode_step[done_v]):
                    #     episode_frames = episode_step * FLAGS.num_action_repeats
                    #
                    #     tf.logging.info('Level: %s Episode return: %f',
                    #                     level_name, episode_return)
                    #
                    #     summary = tf.summary.Summary()
                    #     summary.value.add(tag=level_name + '/episode_return',
                    #                       simple_value=episode_return)
                    #     summary.value.add(tag=level_name + '/episode_frames',
                    #                       simple_value=episode_frames)
                    #     summary_writer.add_summary(summary, num_env_frames_v)
                    #
                    #     if FLAGS.level_name == 'dmlab30':
                    #         level_returns[level_name].append(episode_return)
                    #
                    # if (FLAGS.level_name == 'dmlab30' and
                    #         min(map(len, level_returns.values())) >= 1):
                    #     no_cap = dmlab30.compute_human_normalized_score(level_returns,
                    #                                                     per_level_cap=None)
                    #     cap_100 = dmlab30.compute_human_normalized_score(level_returns,
                    #                                                      per_level_cap=100)
                    #     summary = tf.summary.Summary()
                    #     summary.value.add(
                    #         tag='dmlab30/training_no_cap', simple_value=no_cap)
                    #     summary.value.add(
                    #         tag='dmlab30/training_cap_100', simple_value=cap_100)
                    #     summary_writer.add_summary(summary, num_env_frames_v)
                    #
                    #     # Clear level scores.
                    #     level_returns = {level_name: [] for level_name in level_names}
            else:
                # Execute workers (they just need to enqueue their output).
                while True:
                    session.run(enqueue_ops)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.mode == "train":
        train(7)


if __name__ == '__main__':
    tf.app.run()
