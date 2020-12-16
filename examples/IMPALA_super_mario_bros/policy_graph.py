from ray_helper.miscellaneous import tf_model_ws


def warp_Model():
    import tensorflow as tf
    from infer import categorical
    from module import vtrace_from_logits
    from utils import get_shape

    @tf_model_ws
    class Model(object):
        def __init__(self,
                     act_space,
                     gamma,
                     rnn,
                     use_hrnn,
                     use_reward_prediction,
                     after_rnn,
                     use_pixel_control,
                     use_pixel_reconstruction,
                     is_training=False,
                     **kwargs):
            self.act_space = act_space
            self.use_hrnn = use_hrnn

            self.s = kwargs.get("s")
            self.a = kwargs.get("a")
            self.r = kwargs.get("r")
            self.state_in = kwargs.get("state_in")

            a_onehot = tf.one_hot(
                self.a, depth=act_space, dtype=tf.float32)

            self.feature, self.state_out = self.feature_net(
                self.s, rnn, a_onehot, self.r, self.state_in)

            if self.use_hrnn:
                self.p_zs = self.feature["p_zs"]
                self.p_mus = self.feature["p_mus"]
                self.p_sigmas = self.feature["p_sigmas"]
                self.q_mus = self.feature["q_mus"]
                self.q_sigmas = self.feature["q_sigmas"]
                self.feature = self.feature["q_zs"]

            self.current_act_logits = self.a_net(self.feature)
            self.current_act = tf.squeeze(
                categorical(self.current_act_logits), axis=-1)

            self.current_value = self.v_net(self.feature)

            if is_training:
                self.mask = tf.cast(kwargs.get("mask"), tf.float32)
                self.old_vf = kwargs.get("v_cur")[:, :-1]
                self.behavior_logits = kwargs.get("a_logits")[:, :-1, :]

                current_value = self.current_value * self.mask

                self.current_value, bootstrap_value = current_value[:, :-1], current_value[:, -1]
                feature = self.feature[:, :-1, :]
                self.a = self.a[:, 1:]
                self.r = self.r[:, 1:]
                self.current_act_logits = self.current_act_logits[:, :-1, :]

                vtrace = vtrace_from_logits(
                    self.behavior_logits, self.current_act_logits,
                    self.a, gamma * tf.ones_like(self.a, tf.float32),
                    self.r, self.current_value, bootstrap_value)

                self.vs = vtrace.vs
                self.adv = vtrace.advantages
                self.pg_adv = vtrace.pg_advantages

                self.adv_mean = tf.reduce_mean(self.adv)
                advantages = self.adv - self.adv_mean
                self.adv_std = tf.math.sqrt(tf.reduce_mean(advantages ** 2))

                if use_reward_prediction:
                    if after_rnn:
                        self.reward_prediction = self.r_net(feature)
                    else:
                        raise ValueError("only after rnn")

                if use_pixel_reconstruction:
                    self.pixel_reconstruction = self.reconstruct_net(feature)

                if use_pixel_control:
                    self.pixel_control = self.control_net(feature)

        def get_current_act(self):
            return self.current_act

        def get_current_act_logits(self):
            return self.current_act_logits

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

        def a_net(self, feature, scope="a_net"):
            net = feature
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                net = tf.layers.dense(
                    net,
                    get_shape(feature)[-1],
                    activation=tf.nn.relu,
                    name="dense")
                act_logits = tf.layers.dense(
                    net,
                    self.act_space,
                    activation=None,
                    name="a_logits")

            return act_logits

        def r_net(self, feature, scope="r_net"):
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
                image_feature = tf.reshape(
                    image, [shape[0], shape[1], new_shape[1], new_shape[2], new_shape[3]])

                feature = tf.reshape(
                    image, [shape[0], shape[1], new_shape[1] * new_shape[2] * new_shape[3]])

                cnn_feature = tf.layers.dense(feature, 256, tf.nn.relu, name="feature")
                feature = tf.concat([cnn_feature, prev_a, prev_r[:, :, None]], axis=-1)

                if self.use_hrnn:
                    initial_state = tf.split(state_in, [1, -1], axis=-1)
                    feature, count_out, state_out = rnn(
                        feature, initial_state=initial_state)
                    state_out = tf.concat([count_out, state_out], axis=-1)
                else:
                    initial_state = tf.split(state_in, 2, axis=-1)
                    feature, c_out, h_out = rnn(
                        feature, initial_state=initial_state)
                    state_out = tf.concat([c_out, h_out], axis=-1)

            return feature, state_out

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

        def control_net(self, feature, scope="pixel_control"):
            shape = get_shape(feature)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                feature = tf.reshape(feature, [-1, shape[-1]])
                feature = tf.layers.dense(feature, 7 * 7 * 32, tf.nn.relu, name="feature")
                image = tf.reshape(feature, [-1, 7, 7, 32])
                image = tf.nn.conv2d_transpose(
                    image,
                    filter=tf.get_variable(
                        name="deconv",
                        shape=[9, 9, 32, 32]),
                    output_shape=[get_shape(feature)[0], 21, 21, 32],
                    strides=2,
                    padding="VALID")
                image = tf.nn.relu(image)
                image = tf.nn.conv2d_transpose(
                    image,
                    filter=tf.get_variable(
                        name="control",
                        shape=[4, 4, self.act_space, 32]),
                    output_shape=[get_shape(feature)[0], 21, 21, self.act_space],
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
    frames = kwargs["frames"]
    act_space = kwargs["act_space"]
    gamma = kwargs["gamma"]
    state_size = kwargs["state_size"]
    use_hrnn = kwargs["use_hrnn"]
    use_reward_prediction = kwargs["use_reward_prediction"]
    after_rnn = kwargs["after_rnn"]
    use_pixel_control = kwargs["use_pixel_control"]
    use_pixel_reconstruction = kwargs["use_pixel_reconstruction"]

    phs = dict()

    phs["s"] = tf.placeholder(dtype=tf.uint8, shape=[None, None, 84, 84, frames])
    phs["a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["r"] = tf.placeholder(dtype=tf.float32, shape=[None, None])
    phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

    if use_hrnn:
        rnn = TmpHierRNN(4, 64, 4, 2, 8, 'lstm', 'rmc',
                         return_sequences=True, return_state=True, name="hrnn")
    else:
        rnn = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")

    model = Model(act_space, gamma, rnn, use_hrnn,
                  use_reward_prediction, after_rnn, use_pixel_control,
                  use_pixel_reconstruction, False, **phs)

    return model
