# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from XtoBeREMOVED.REINFORCE import REINFORCE


class PPOc(REINFORCE):
    """
    https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self,
                 sess,
                 model,
                 optimizer,
                 epsilon,
                 a0=1.0,
                 gamma=1.0,
                 entropy=0.01,
                 actor_freeze_update_per_steps=None,
                 actor_freeze_update_step_size=1.0,
                 trace_decay_rate=None,
                 icm=None,
                 grad_norm_clip=None,
                 scope="PPO_clip"):
        """
        PPO_penalty without Critic,
            for the case when actor & critic DO NOT share variables,
            it gives more flexibility to separate actor & critic

        epsilon:
            controls the clip region of ppo,
            where the target function of ppo_clip is
            E[min(r * A, clip(r, 1-epsilon, 1+epsilon) * A)]
        """
        self.epsilon = epsilon
        self.a0 = a0
        self.a_freeze_update_per_steps = actor_freeze_update_per_steps
        self.a_freeze_update_step_size = actor_freeze_update_step_size

        super().__init__(
            sess=sess,
            model=model,
            optimizer=optimizer,
            off_policy=False,
            gamma=gamma,
            entropy=entropy,
            trace_decay_rate=trace_decay_rate,
            icm=icm,
            grad_norm_clip=grad_norm_clip,
            scope=scope)

    def _build_loss_fn(self):
        """
        A_t:
            return or advantage or td error, etc, shape = [None, (*,)]
        old_act_probs:
            real_act act prob of old policy, shape = [None, (*,) act_nums]
        """
        act_probs = self.model.get_current_act_probs()
        old_act_probs = tf.stop_gradient(
            self.model.get_old_act_probs())

        shape = self.get_shape(act_probs)
        act_nums = shape[-1]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.A_t = self.model.get_advantage()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        act_onehot = tf.one_hot(self.model.a_t, depth=act_nums)

        ro_t = tf.reduce_sum(
            act_onehot * act_probs, axis=-1
        ) / tf.maximum(tf.reduce_sum(
            act_onehot * old_act_probs,
            axis=-1), 1e-8)
        ro_t_clip = tf.clip_by_value(
            ro_t, 1.0 - self.epsilon, 1.0 + self.epsilon)
        clip_loss = tf.minimum(ro_t * self.A_t, ro_t_clip * self.A_t)
        self.a_loss = tf.reduce_mean(
            self.get_slots(
                self.build_icm_loss(
                    self.build_entropy_loss(
                        - self.a0 * clip_loss))))

        tf.summary.scalar("a_loss_clip", tf.reduce_mean(self.get_slots(-clip_loss)))
        tf.summary.scalar("a_loss_all", self.a_loss)

    def _build_reset_op(self):
        super()._build_reset_op()

    def _build_train_op(self):
        grads_and_vars = self.compute_gradient(
            self.a_loss, self.model.get_actor_current_variables())
        self.a_current_train_op = self.apply_gradients(grads_and_vars)

        # self.actor_ovars_to_cvars = OrderedDict()
        # self.assert_variables(self.model.get_actor_old_variables(),
        #                       self.model.get_actor_current_variables(),
        #                       self.actor_ovars_to_cvars)

        self.a_old_train_op = []
        # for o_var, c_var in self.actor_ovars_to_cvars.items():
        #     self.a_old_train_op.append(
        #         tf.assign_add(
        #             o_var,
        #             self.a_freeze_update_step_size * (c_var - o_var)))
        #     self.init_target_op.append(tf.assign(o_var, c_var))

    def update(self, _global_step, feed_dict):
        _, loss = self.sess.run([self.a_current_train_op, self.a_loss], feed_dict)

        # if _global_step == 0:
        #     self.sess.run(self.init_target_op)

        if self.a_freeze_update_per_steps is not None:
            if _global_step % self.a_freeze_update_per_steps:
                self.sess.run(self.a_old_train_op)

        return loss
