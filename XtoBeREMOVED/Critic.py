# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict

from XtoBeREMOVED.base_class import base


class Critic(base):
    def __init__(self,
                 sess,
                 model,
                 optimizer,
                 c0=1.0,
                 use_freeze=True,
                 critic_freeze_update_per_steps=8,
                 critic_freeze_update_step_size=0.01,
                 gamma=1.0,
                 entropy=None,
                 trace_decay_rate=None,
                 icm=None,
                 popart_step_size=None,
                 grad_norm_clip=None,
                 scope="Critic"):
        """
        This is the base of all value-based algorithms.
        No policy-based thing is included in this class.
        No MC is included in this class.
        This class supports TD-based algorithm,
            for instance, TD, SARSA, Q
        Not Support tpu or estimators.

        model: model architecture function,
            many modules need to be constructed in model,
            which includes
                .get_variables
                    this method will return all the variables that need to
                    be traced if eligibility trace is employed
                .get_target_value
                    when calculating n_steps advantage/td_error,
                    r_t_0 + gamma * r_t_1 + ... + gamma^(n-1) * r_t_(n-1)
                    + gamma^n * target_value(s_t_n) - current_value(s_t_0),
                    target_value(s_t_n) is needed.
                    This method returns exactly target_value (type: tensor).
                .get_current_value
                    this is the current value
                .get_critic_target_variables
                    the variables for target_value
                    name = "target/.*" for instance
                .get_critic_current_variables
                    the variables for current_value
                    name = "current/.*" for instance
                (.get_last_layer_vars)
                    this method will be called when use_PopArt == True
                    will return (w, b)

        use_freeze:
            whether to use freeze trick
            for general, target function is
                (r + gamma * stop_grad(A(s_t+1)) - A(s))^2
            the freeze trick replace A(s_t+1) by A_old(s_t+1)
                and update A_old by exponential moving average,
                which is A_old <- A_old + alpha * (A - A_old)
        freeze_update_per_steps:
            updating A_old for how many steps
        freeze_update_step_size:
            alpha

        popart_step_size:
            Not using if None
            the step size for update mean and variance
        """
        self.use_freeze = use_freeze
        self.c0 = c0
        self.c_freeze_update_per_steps = critic_freeze_update_per_steps
        self.c_freeze_update_step_size = critic_freeze_update_step_size
        self.popart_step_size = popart_step_size

        super().__init__(
            sess=sess,
            model=model,
            optimizer=optimizer,
            gamma=gamma,
            entropy=entropy,
            trace_decay_rate=trace_decay_rate,
            icm=icm,
            grad_norm_clip=grad_norm_clip,
            scope=scope)

    def _build_loss_fn(self):
        ret = getattr(self.model, "get_return", None)
        if ret is not None:
            self.target = ret()
        else:
            self.target = self.build_target(self.model.get_target_value())

        # this advantage term represents exactly \
        # the term before \grad ln pi(a|s) in REINFORCE.
        v = self.model.get_current_value()
        if v.__class__ is not list:
            v = [v]
        self.advantage = self.target - v[0]
        losses = [tf.square(self.target - vi) for vi in v]

        self.c_loss = tf.reduce_mean(
            self.get_slots(
                self.build_icm_loss(
                    self.build_entropy_loss(
                        self.c0 * tf.reduce_max(losses, axis=0))))) / 2.

        tf.summary.scalar("c_loss_clip", tf.reduce_mean(self.get_slots(tf.reduce_max(losses, axis=0))) / 2.)
        tf.summary.scalar("c_loss_all", self.c_loss)

    def _build_reset_op(self):
        super()._build_reset_op()

    def _build_train_op(self):
        grads_and_vars = self.compute_gradient(
            self.c_loss, self.model.get_critic_current_variables())
        self.c_current_train_op = self.apply_gradients(grads_and_vars)

        self.c_target_train_op = []
        if self.use_freeze:
            self.target_vars_to_current_vars = OrderedDict()
            self.assert_variables(self.model.get_critic_target_variables(),
                                  self.model.get_critic_current_variables(),
                                  self.target_vars_to_current_vars)

            for t_var, c_var in self.target_vars_to_current_vars.items():
                self.c_target_train_op.append(
                    tf.assign_add(
                        t_var,
                        self.c_freeze_update_step_size * (c_var - t_var)))
                self.init_target_op.append(tf.assign(t_var, c_var))

        self.popart_train_op = self.build_popart_train_op()

    def update(self, _global_step, feed_dict):
        _, __, advantage, loss = self.sess.run(
            [self.c_current_train_op,
             self.popart_train_op,
             self.advantage,
             self.c_loss],
            feed_dict)

        if _global_step == 0:
            self.sess.run(self.init_target_op)

        if self.c_freeze_update_per_steps is not None:
            if _global_step % self.c_freeze_update_per_steps == 0:
                self.sess.run(self.c_target_train_op)

        return advantage, loss

    def act(self, feed_dict):
        return self.sess.run(self.model.get_current_act(), feed_dict)

    def build_target(self, target_value):
        shape = self.get_shape(target_value)
        r_shape = [t if t.__class__ == int else None for t in shape]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.r_t = tf.placeholder(
            dtype=tf.float32, shape=r_shape + [None], name="n_steps_rewards")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        n_steps = self.get_shape(self.r_t)[1]
        powers = tf.linspace(0., tf.cast(n_steps - 1, tf.float32), n_steps)
        discounts = tf.pow(self.gamma, powers)
        for i in range(len(shape)):
            discounts = tf.expand_dims(discounts, axis=0)
        target1 = tf.reduce_sum(self.r_t * discounts, axis=-1)

        if self.popart_step_size is None:
            target2 = tf.pow(self.gamma, tf.cast(n_steps, tf.float32)
                             ) * tf.stop_gradient(target_value)
            target = target1 + target2
        else:
            self.popart_mu = tf.get_variable(
                name="popart/mu",
                shape=[],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=False)
            self.popart_var = tf.get_variable(
                name="popart/var",
                shape=[],
                dtype=tf.float32,
                initializer=tf.ones_initializer(),
                trainable=False)
            self.popart_sigma = tf.sqrt(
                self.popart_var - tf.square(self.popart_mu))

            target2 = tf.pow(
                self.gamma, tf.cast(n_steps, tf.float32)
            ) * tf.stop_gradient(
                target_value * self.popart_sigma + self.popart_mu)
            target = target1 + target2

            target = (target - self.popart_mu) / self.popart_sigma

        return target

    def build_popart_train_op(self):
        popart_train_op = []
        if self.popart_step_size is not None:
            avg_target = tf.reduce_mean(self.target)
            new_popart_mu = self.popart_mu + self.popart_step_size * (
                    avg_target * self.popart_sigma + self.popart_mu - self.popart_mu)
            new_popart_var = self.popart_var + self.popart_step_size * (
                    tf.square(
                        avg_target * self.popart_sigma + self.popart_mu
                    ) - self.popart_var)
            new_popart_sigma = tf.sqrt(new_popart_var - tf.square(new_popart_mu))

            op = getattr(self.model, "get_last_layer_vars", None)
            if callable(op):
                w, b = op()
                popart_train_op.append(
                    tf.assign(w, self.popart_sigma / new_popart_sigma * w))
                popart_train_op.append(
                    tf.assign(
                        b,
                        (self.popart_sigma * b + self.popart_mu
                         - new_popart_mu) / new_popart_sigma))
            popart_train_op.append(tf.assign(self.popart_mu, new_popart_mu))
            popart_train_op.append(tf.assign(self.popart_var, new_popart_var))
        return popart_train_op
