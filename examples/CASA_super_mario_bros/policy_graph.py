from ray_helper.miscellaneous import tf_model_ws


def warp_Model():
    import tensorflow as tf
    from infer import categorical
    from module import duelingQ
    from module import doubleQ
    from module import vtrace_from_logits
    from module import retrace_from_logits
    from utils import get_shape

    @tf_model_ws
    class Model(object):
        def __init__(self,
                     act_space,
                     gamma,
                     n_step,
                     rnn,
                     use_hrnn,
                     use_rmc,
                     use_amc,
                     use_beta,
                     use_reward_prediction,
                     after_rnn,
                     use_pixel_control,
                     is_training=False,
                     **kwargs):
            self.act_space = act_space
            self.n_step = n_step
            self.use_hrnn = use_hrnn
            self.use_rmc = use_rmc
            self.use_amc = use_amc

            self.s = kwargs.get("s")
            self.a = kwargs.get("a")
            self.r = kwargs.get("r")
            self.state_in = kwargs.get("state_in")

            feature, self.state_out = self.feature_net(
                self.s, rnn, self.a, self.r, self.state_in)

            if self.use_hrnn:
                self.p_zs = feature["p_zs"]
                self.p_mus = feature["p_mus"]
                self.p_sigmas = feature["p_sigmas"]
                self.q_mus = feature["q_mus"]
                self.q_sigmas = feature["q_sigmas"]
                feature = feature["q_zs"]

            with tf.variable_scope("alpha", reuse=tf.AUTO_REUSE):
                alpha = tf.get_variable(name="alpha",
                                        shape=(1, 1, 1),
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
            tf.summary.scalar("alpha", tf.reduce_mean(alpha))
            alpha = tf.log(1.0 + tf.exp(alpha))

            self.qf, self.current_value, self.current_act_logits = self.q_fn(feature, alpha, use_beta, "q")

            self.current_act = tf.squeeze(
                categorical(self.current_act_logits), axis=-1)

            if is_training:
                self.mask = tf.cast(kwargs.get("mask"), tf.float32)
                self.behavior_logits = kwargs.get("a_logits")
                self.old_vf = kwargs.get("v_cur")
                self.current_value = self.current_value * self.mask

                '''
                get qa & qa1 & n_step_rewards
                '''
                self.qa = tf.reduce_sum(
                    tf.one_hot(
                        self.a[:, 1: 1 - self.n_step],
                        depth=self.act_space, dtype=tf.float32
                    ) * self.qf[:, :-n_step], axis=-1) * self.mask[:, :-n_step]

                self.qf1, _, _ = self.q_fn(feature, alpha, use_beta, "q_target")
                q1f = self.qf[:, n_step:, :]
                q1f1 = self.qf1[:, n_step:, :]
                self.qa1 = doubleQ(q1f1, q1f) * self.mask[:, n_step:]
                # self.q1f = self.qf[:, n_step:, :]
                # self.qa1 = tf.reduce_max(self.q1f, axis=-1) * self.mask[:, n_step:]

                gammas = tf.pow(
                    gamma, tf.range(0, get_shape(self.r)[1], dtype=tf.float32))
                gammas_1 = 1.0 / gammas
                returns = tf.cumsum(self.r * gammas[None, :], axis=1)
                discount_n_step_rewards = returns[:, n_step:] - returns[:, :-n_step]
                self.n_step_rewards = discount_n_step_rewards * gammas_1[None, :-n_step]

                self.n_step_qs = tf.stop_gradient(
                    self.n_step_rewards + gamma ** n_step * self.qa1)

                # target_values = tf.reduce_sum(
                #     tf.one_hot(
                #         self.a[:, 1: 1 - self.n_step],
                #         depth=self.act_space, dtype=tf.float32
                #     ) * self.qf1[:, :-n_step], axis=-1) * self.mask[:, :-n_step]

                retrace = retrace_from_logits(
                    self.behavior_logits[:, :-n_step, :],
                    self.current_act_logits[:, :-n_step, :],
                    self.a[:, 1:1 - n_step],
                    gamma * tf.ones_like(self.a[:, 1:1 - n_step], tf.float32),
                    tf.ones_like(self.a[:, 1:1 - n_step], tf.float32),
                    self.r[:, 1:1 - n_step],
                    self.qa, self.qa, self.qa1[:, -n_step])

                self.retrace_qs = retrace.qs

                '''
                get vtrace
                '''
                vtrace = vtrace_from_logits(
                    self.behavior_logits[:, :-n_step, :], self.current_act_logits[:, :-n_step, :],
                    self.a[:, 1:1 - n_step], gamma * tf.ones_like(self.a[:, 1:1 - n_step], tf.float32),
                    self.r[:, 1:1 - n_step], self.current_value[:, :-n_step], self.current_value[:, -n_step])

                self.vs = vtrace.vs
                self.adv = vtrace.advantages
                self.pg_adv = vtrace.pg_advantages

                self.adv_mean = tf.reduce_mean(self.adv)
                advantages = self.adv - self.adv_mean
                self.adv_std = tf.math.sqrt(tf.reduce_mean(advantages ** 2))

                if use_reward_prediction:
                    if after_rnn:
                        self.reward_prediction = self.r_net(feature[:, :-n_step, :])
                    else:
                        raise ValueError("only after rnn")

                if use_pixel_control:
                    self.pixel_control = self.control_net(feature[:, :-n_step, :])

        def get_current_act(self):
            return self.current_act

        def q_fn(self, feature, alpha, use_beta, scope):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                vf = self.v_net(feature)
                adv = self.adv_net(feature)
                adv = adv - tf.reduce_mean(adv, axis=-1, keepdims=True)
                if use_beta:
                    p = tf.stop_gradient(tf.nn.softmax(alpha * adv, axis=-1))
                    E_adv = tf.reduce_sum(adv * p, axis=-1, keepdims=True)
                    qf = tf.expand_dims(vf, axis=-1) + adv - E_adv
                else:
                    qf = tf.expand_dims(vf, axis=-1) + adv
            return qf, vf, alpha * adv

        def v_net(self, feature, scope="v_net"):
            net = feature
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tf.layers.dense(
                    net,
                    get_shape(feature)[-1],
                    activation=tf.nn.relu,
                    name="dense")
                v_value = tf.squeeze(
                    tf.layers.dense(
                        net,
                        1,
                        activation=None,
                        name="v_value"),
                    axis=-1)
            return v_value

        def adv_net(self, feature, scope="adv_net"):
            net = feature
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tf.layers.dense(
                    net,
                    get_shape(feature)[-1],
                    activation=tf.nn.relu,
                    name="dense")
                adv = tf.layers.dense(
                    net,
                    self.act_space,
                    activation=None,
                    name="q_value")
            return adv

        def feature_net(self, image, rnn, a, r, state_in, scope="feature"):
            shape = get_shape(image)
            image = tf.cast(image, tf.float32) / 255.0
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
                feature = tf.reshape(
                    image, [shape[0], shape[1], new_shape[1] * new_shape[2] * new_shape[3]])

                a_onehot = tf.one_hot(
                    a, depth=self.act_space, dtype=tf.float32)

                feature = tf.layers.dense(feature, 256, tf.nn.relu, name="feature")
                feature = tf.concat([feature, a_onehot, r[:, :, None]], axis=-1)

                if self.use_hrnn:
                    initial_state = tf.split(state_in, [1, -1], axis=-1)
                    feature, count_out, state_out = rnn(
                        feature, initial_state=initial_state)
                    state_out = tf.concat([count_out, state_out], axis=-1)
                elif self.use_rmc:
                    initial_state = [state_in]
                    feature, state_out = rnn(
                        feature, initial_state=initial_state)
                elif self.use_amc:
                    initial_state = tf.split(state_in, [1, 64, 64, -1], axis=-1)
                    feature, count_out, ns_out, pos_out, state_out = rnn(
                        feature, initial_state=initial_state)
                    state_out = tf.concat([count_out, ns_out, pos_out, state_out], axis=-1)
                else:
                    c_in, h_in = tf.split(state_in, 2, axis=-1)
                    feature, c_out, h_out = rnn(
                        feature, initial_state=[c_in, h_in])
                    state_out = tf.concat([c_out, h_out], axis=-1)

            return feature, state_out

        def r_net(self, feature, scope="r_net"):
            net = feature
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tf.layers.dense(
                    net,
                    get_shape(feature)[-1],
                    activation=tf.nn.relu,
                    name="dense")
                r = tf.squeeze(
                    tf.layers.dense(
                        net,
                        1,
                        activation=None,
                        name="r"),
                    axis=-1)
            return r

        def control_net(self, feature, scope="control"):
            shape = get_shape(feature)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                feature = tf.layers.dense(feature, 7 * 7 * 32, tf.nn.relu, name="feature")
                with tf.variable_scope("pixel", reuse=tf.AUTO_REUSE):
                    image = tf.reshape(feature, [-1, 7, 7, 32])
                    image = tf.nn.conv2d_transpose(
                        image,
                        filter=tf.get_variable(
                            name="deconv0",
                            shape=[9, 9, 32, 32]),
                        output_shape=[get_shape(image)[0], 21, 21, 32],
                        strides=2,
                        padding="VALID")
                    image = tf.nn.relu(image)
                    image = tf.nn.conv2d_transpose(
                        image,
                        filter=tf.get_variable(
                            name="deconv1",
                            shape=[4, 4, get_shape(self.s)[-1], 32]),
                        output_shape=[get_shape(image)[0], 21, 21, get_shape(self.s)[-1]],
                        strides=1,
                        padding="SAME")
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

    return Model


def build_evaluator_model(kwargs):
    Model = warp_Model()
    import tensorflow as tf
    from module import TmpHierRNN
    from module import AMCRNN
    from module import RMCRNN
    from module import NewLSTM

    frames = kwargs["frames"]
    image_size = kwargs["image_size"]
    act_space = kwargs["act_space"]
    gamma = kwargs["gamma"]
    n_step = kwargs["n_step"]
    time_scale = kwargs["time_scale"]
    state_size = kwargs["state_size"]
    use_hrnn = kwargs["use_hrnn"]
    use_rmc = kwargs["use_rmc"]
    use_amc = kwargs["use_amc"]
    use_beta = kwargs["use_beta"]
    use_reward_prediction = kwargs["use_reward_prediction"]
    after_rnn = kwargs["after_rnn"]
    use_pixel_control = kwargs["use_pixel_control"]

    phs = dict()

    phs["s"] = tf.placeholder(
        dtype=tf.uint8, shape=[None, None, image_size, image_size, frames])
    phs["a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["r"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

    if use_hrnn:
        rnn = TmpHierRNN(time_scale, 64, 4, 2, 8, 'lstm', 'rmc',
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
        rnn = NewLSTM(
            256, return_sequences=True, return_state=True, name="lstm")

    model = Model(act_space, gamma, n_step, rnn, use_hrnn, use_rmc, use_amc, use_beta, use_reward_prediction,
                  after_rnn, use_pixel_control, False, **phs)

    return model
