from ray_helper.miscellaneous import tf_model_ws


def warp_Model():
    import tensorflow as tf
    from infer import categorical
    from utils import get_shape

    @tf_model_ws
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

        def a_net(self, feature, scope):
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

    return Model


def build_evaluator_model(kwargs):
    Model = warp_Model()
    import tensorflow as tf
    from module import RMCRNN
    from module import TmpHierRMCRNN_v2
    frames = kwargs["frames"]
    act_space = kwargs["act_space"]
    state_size = kwargs["state_size"]
    use_rmc = kwargs["use_rmc"]
    use_hrmc = kwargs["use_hrmc"]
    use_reward_prediction = kwargs["use_reward_prediction"]
    after_rnn = kwargs["after_rnn"]
    use_pixel_control = kwargs["use_pixel_control"]

    phs = dict()

    phs["s"] = tf.placeholder(dtype=tf.float32, shape=[None, None, 84, 84, 3 * frames])
    phs["prev_a"] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    phs["prev_r"] = tf.placeholder(dtype=tf.float32, shape=[None, None])

    if use_hrmc:
        phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        rnn = TmpHierRMCRNN_v2(
            4, 64, 4, 4, 8, return_sequences=True, return_state=True, name="hrmcrnn")
    elif use_rmc:
        phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        rnn = RMCRNN(
            64, 4, 4, return_sequences=True, return_state=True, name="rmcrnn")
    else:
        phs["state_in"] = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
        rnn = tf.compat.v1.keras.layers.LSTM(
            256, return_sequences=True, return_state=True, name="lstm")

    model = Model(act_space, rnn, use_rmc, use_hrmc,
                  use_reward_prediction, after_rnn,
                  use_pixel_control, "agent", **phs)

    return model