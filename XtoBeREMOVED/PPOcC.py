# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from XtoBeREMOVED.PPOc import PPOc
from XtoBeREMOVED.Critic import Critic


class PPOcCritic(PPOc, Critic):
    """
    https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self,
                 sess,
                 model,
                 optimizer,
                 epsilon,
                 a0=1.0,
                 c0=1.0,
                 gamma=1.0,
                 entropy=0.01,
                 actor_freeze_update_per_steps=None,
                 actor_freeze_update_step_size=1.0,
                 use_freeze=False,
                 critic_freeze_update_per_steps=8,
                 critic_freeze_update_step_size=0.01,
                 trace_decay_rate=None,
                 icm=None,
                 popart_step_size=None,
                 grad_norm_clip=None,
                 scope="PPOcC"):
        """
        PPO_clip with Critic,
            for the case when actor & critic share variables,
            it's stable to update variables simultaneously

        epsilon:
            controls the clip region of ppo,
            where the target function of ppo_clip is
            E[min(r * A, clip(r, 1-epsilon, 1+epsilon) * A)]
        """
        with tf.variable_scope(scope):
            self.a0 = a0
            self.c0 = c0

            self.sess = sess
            self.model = model
            self.optimizer = optimizer
            self.epsilon = epsilon
            self.gamma = gamma
            self.entropy = entropy
            if entropy is not None:
                self.entropy = entropy * .5
            self.actor_freeze_update_per_steps = actor_freeze_update_per_steps
            self.actor_freeze_update_step_size = actor_freeze_update_step_size
            self.use_freeze = use_freeze
            self.critic_freeze_update_per_steps = critic_freeze_update_per_steps
            self.critic_freeze_update_step_size = critic_freeze_update_step_size
            self.trace_decay_rate = trace_decay_rate
            self.icm = icm
            if icm is not None:
                self.icm = icm * .5
            self.popart_step_size = popart_step_size
            self.grad_norm_clip = grad_norm_clip

            self.build_init_target_op()
            self.build_trace()

            self._build_loss_fn()
            self._build_reset_op()
            self._build_train_op()

    def _build_loss_fn(self):
        Critic._build_loss_fn(self)
        PPOc._build_loss_fn(self)

        self.ac_loss = self.a_loss + self.c_loss

        tf.summary.scalar("ac_loss", self.ac_loss)

    def _build_reset_op(self):
        PPOc._build_reset_op(self)

    def _build_train_op(self):
        # Critic._build_train_op(self)
        # PPOc._build_train_op(self)

        grads_and_vars = self.compute_gradient(
            self.ac_loss,
            list(set(
                self.model.get_actor_current_variables()
                + self.model.get_critic_current_variables())),
            summary=True)
        self.ac_current_train_op = self.apply_gradients(grads_and_vars)

    def update(self, _global_step, feed_dict):
        _, __, advantage, a_loss, c_loss, ac_loss = self.sess.run(
            [self.ac_current_train_op,
             self.popart_train_op,
             self.advantage,
             self.a_loss,
             self.c_loss,
             self.ac_loss],
            feed_dict)

        if _global_step == 0:
            self.sess.run(self.init_target_op)

        if self.a_freeze_update_per_steps is not None:
            if _global_step % self.a_freeze_update_per_steps:
                self.sess.run(self.a_old_train_op)

        if self.c_freeze_update_per_steps is not None:
            if _global_step % self.c_freeze_update_per_steps == 0:
                self.sess.run(self.c_target_train_op)

        return advantage, a_loss, c_loss, ac_loss
