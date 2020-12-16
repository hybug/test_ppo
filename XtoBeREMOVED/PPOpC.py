# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from XtoBeREMOVED.PPOp import PPOp
from XtoBeREMOVED.Critic import Critic


class PPOpCritic(PPOp, Critic):
    """
    https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self,
                 sess,
                 model,
                 optimizer,
                 d_target,
                 c0=1.0,
                 gamma=1.0,
                 entropy=0.01,
                 actor_freeze_update_per_steps=None,
                 actor_freeze_update_step_size=1.0,
                 use_freeze=True,
                 critic_freeze_update_per_steps=8,
                 critic_freeze_update_step_size=0.01,
                 trace_decay_rate=None,
                 icm=None,
                 popart_step_size=None,
                 grad_norm_clip=None,
                 scope="PPOpC"):
        """
        PPO_penalty with Critic,
            for the case when actor & critic share variables,
            it's stable to update variables simultaneously

        d_target: controls the KL_penalty level,
            where the target function of ppo_penalty is
            E[r * A - beta * KL(old_act_probs | act_probs)],
            where if KL < d_target / 1.5, beta <- beta / 2,
                  if KL > d_target * 1.5, beta <- beta * 2
        """
        self.c0 = c0

        with tf.variable_scope(scope):
            self.sess = sess
            self.model = model
            self.optimizer = optimizer
            self.d_target = d_target
            self.gamma = gamma
            self.entropy = entropy * .5
            self.actor_freeze_update_per_steps = actor_freeze_update_per_steps
            self.actor_freeze_update_step_size = actor_freeze_update_step_size
            self.use_freeze = use_freeze
            self.critic_freeze_update_per_steps = critic_freeze_update_per_steps
            self.critic_freeze_update_step_size = critic_freeze_update_step_size
            self.trace_decay_rate = trace_decay_rate
            self.icm = icm
            self.popart_step_size = popart_step_size
            self.grad_norm_clip = grad_norm_clip

            self.build_init_target_op()
            self.build_trace()

            self._build_loss_fn()
            self._build_reset_op()
            self._build_train_op()

    def _build_loss_fn(self):
        Critic._build_loss_fn(self)
        PPOp._build_loss_fn(self)

        self.ac_loss = self.a_loss + self.c0 * self.c_loss

    def _build_reset_op(self):
        PPOp._build_reset_op(self)

    def _build_train_op(self):
        Critic._build_train_op(self)
        PPOp._build_train_op(self)

        grads_and_vars = self.compute_gradient(
            self.ac_loss,
            list(set(
                self.model.get_actor_current_variables()
                + self.model.get_critic_current_variables())))
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
