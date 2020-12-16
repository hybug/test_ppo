# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict

from XtoBeREMOVED.DDPG import DDPG


class TD3(DDPG):
    """
    https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self,
                 sess,
                 model,
                 optimizer,
                 actor_update_per_steps=8,
                 actor_freeze_update_per_steps=8,
                 actor_freeze_update_step_size=0.01,
                 critic_freeze_update_per_steps=1,
                 critic_freeze_update_step_size=0.01,
                 gamma=1.0,
                 entropy=None,
                 trace_decay_rate=None,
                 icm=None,
                 popart_step_size=None,
                 grad_norm_clip=None,
                 scope="TD3"):
        """
        model: model architecture function,
            many modules need to be constructed in model,
            which includes
                .get_current_act
                    since td3 is a continuous control
                    algorithm with deterministic policy,
                    this gives the action with a white noise
                        a + noise = pi(s) + noise
                .get_current_exact_act
                    this gives the exact center pi(s)

                .get_actor_target_variables
                    target variables for actor
                    name = "atarget/.*" for instance
                .get_actor_current_variables
                    current variables for actor
                    name = "acurrent/.*" for instance

                .get_target1_value
                    critic1(s_t1, actor(s_t1) + epsilon)
                .get_current1_value
                    critic1(s_t, a_t)
                .get_current1_exact_value
                    critic1(s_t, actor(s_t))

                .get_target2_value
                    critic2(s_t1, actor(s_t1) + epsilon)
                .get_current2_value
                    critic2(s_t, a_t)
                .get_current2_exact_value
                    critic2(s_t, actor(s_t))

                .get_critic1_target_variables
                    the variables for target_value
                    name = "c1target/.*" for instance
                .get_critic1_current_variables
                    the variables for current_value
                    name = "c1current/.*" for instance
                .get_critic2_target_variables
                    the variables for target_value
                    name = "c2target/.*" for instance
                .get_critic2_current_variables
                    the variables for current_value
                    name = "c2current/.*" for instance

                (.get_last_layer_vars)
                    this method will be called when use_PopArt == True
                    will return (w, b)
        """
        super().__init__(
            sess=sess,
            model=model,
            optimizer=optimizer,
            actor_update_per_steps=actor_update_per_steps,
            actor_freeze_update_per_steps=actor_freeze_update_per_steps,
            actor_freeze_update_step_size=actor_freeze_update_step_size,
            critic_freeze_update_per_steps=critic_freeze_update_per_steps,
            critic_freeze_update_step_size=critic_freeze_update_step_size,
            gamma=gamma,
            entropy=entropy,
            trace_decay_rate=trace_decay_rate,
            icm=icm,
            popart_step_size=popart_step_size,
            grad_norm_clip=grad_norm_clip,
            scope=scope)

    def _build_loss_fn(self):
        self.target = self.build_target(
            tf.reduce_min(
                [self.model.get_target1_value(),
                 self.model.get_target2_value()],
                axis=0))

        v = self.model.get_current1_value()
        if v.__class__ is not list:
            v = [v]
        self.advantage1 = self.target - v[0]
        losses1 = [tf.square(self.target - vi) for vi in v]

        self.c1_loss = tf.reduce_mean(
            self.get_slots(
                self.build_icm_loss(
                    tf.reduce_max(losses1, axis=0)))) / 2.

        v = self.model.get_current2_value()
        if v.__class__ is not list:
            v = [v]
        self.advantage2 = self.target - v[0]
        losses2 = [tf.square(self.target - vi) for vi in v]

        self.c2_loss = tf.reduce_mean(
            self.get_slots(
                tf.reduce_max(losses2))) / 2.

        self.a_loss = tf.reduce_mean(
            self.get_slots(
                self.build_entropy_loss(
                    - self.model.get_current1_exact_value())))

    def _build_reset_op(self):
        super()._build_reset_op()

    def _build_train_op(self):
        self.actor_tvars_to_cvars = OrderedDict()
        self.assert_variables(self.model.get_actor_target_variables(),
                              self.model.get_actor_current_variables(),
                              self.actor_tvars_to_cvars)

        self.critic1_tvars_to_cvars = OrderedDict()
        self.assert_variables(self.model.get_critic1_target_variables(),
                              self.model.get_critic1_current_variables(),
                              self.critic1_tvars_to_cvars)

        self.critic2_tvars_to_cvars = OrderedDict()
        self.assert_variables(self.model.get_critic2_target_variables(),
                              self.model.get_critic2_current_variables(),
                              self.critic2_tvars_to_cvars)

        grads_and_vars = self.compute_gradient(
            self.c1_loss, self.model.get_critic1_current_variables())
        self.c1_current_train_op = self.apply_gradients(grads_and_vars)

        grads_and_vars = self.compute_gradient(
            self.c2_loss, self.model.get_critic2_current_variables())
        self.c2_current_train_op = self.apply_gradients(grads_and_vars)

        grads_and_vars = self.compute_gradient(
            self.a_loss, self.model.get_actor_current_variables())
        self.a_current_train_op = self.apply_gradients(grads_and_vars)

        self.c1_target_train_op = []
        for t_var, c_var in self.critic1_tvars_to_cvars.items():
            self.c1_target_train_op.append(
                tf.assign_add(
                    t_var,
                    self.c_freeze_update_step_size * (c_var - t_var)))
            self.init_target_op.append(tf.assign(t_var, c_var))

        self.c2_target_train_op = []
        for t_var, c_var in self.critic2_tvars_to_cvars.items():
            self.c2_target_train_op.append(
                tf.assign_add(
                    t_var,
                    self.c_freeze_update_step_size * (c_var - t_var)))
            self.init_target_op.append(tf.assign(t_var, c_var))

        self.a_target_train_op = []
        for t_var, c_var in self.actor_tvars_to_cvars.items():
            self.a_target_train_op.append(
                tf.assign_add(
                    t_var,
                    self.a_freeze_update_step_size * (c_var - t_var)))
            self.init_target_op.append(tf.assign(t_var, c_var))

        self.popart_train_op = self.build_popart_train_op()

    def update(self, _global_step, feed_dict):
        _, __, advantage1, c1_loss = self.sess.run(
            [self.c1_current_train_op,
             self.popart_train_op,
             self.advantage1,
             self.c1_loss],
            feed_dict)

        _, advantage2, c2_loss = self.sess.run(
            [self.c2_current_train_op, self.advantage2, self.c2_loss], feed_dict)

        if _global_step == 0:
            self.sess.run(self.init_target_op)

        if _global_step % self.a_update_per_steps == 0:
            _, a_loss = self.sess.run(
                [self.a_current_train_op, self.a_loss], feed_dict)

        if self.c_freeze_update_per_steps is not None:
            if _global_step % self.c_freeze_update_per_steps == 0:
                self.sess.run(self.c1_target_train_op)
                self.sess.run(self.c2_target_train_op)

        if self.a_freeze_update_per_steps is not None:
            if _global_step % self.a_freeze_update_per_steps == 0:
                self.sess.run(self.a_target_train_op)

        return advantage1, c1_loss
