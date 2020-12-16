# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from XtoBeREMOVED.base_class import base


class REINFORCE(base):
    """
    </RL-an introduction/> 2nd ed page 275
    """

    def __init__(self,
                 sess,
                 model,
                 optimizer,
                 off_policy=True,
                 gamma=1.0,
                 entropy=0.01,
                 trace_decay_rate=None,
                 icm=None,
                 grad_norm_clip=None,
                 scope="REINFORCE"):
        """
        This is the base of all policy-based algorithms.
        No value-based thing is included in this class,
        If Actor-Critic is needed, please feed advantages
            (G_t - V_t for instance) to A_t as well as
            updating V_t additionally.
        If average reward is need, similarly, feed (r_t - r_avg_t) to A_t
        Not Support tpu or estimators.

        model:
            model architecture function,
            many modules need to be constructed in model,
            which includes
                .get_current_act
                    the sampled action that the model acts, multi-head
                .get_current_act_probs
                    the action distribution, multi-head
                .get_current_act_logits
                    the logits of act prob distribution, multi-head
                .get_variables
                    the variables that should be used for trace,
                    please notice that this is not always
                    tf.all_variables or tf.trainable_variables

        off_policy:
            how to train the model, on-policy or off-policy
            if off_policy, using importance sampling


        """
        self.off_policy = off_policy
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
        """
        A_t:
            return or advantage or td error, etc, shape = [None, (*,)]
        """
        act_probs = self.model.get_act_probs()
        shape = self.get_shape(act_probs)
        act_nums = shape[-1]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.A_t = self.model.get_advantage()
        if self.off_policy:
            self.old_act_probs = self.model.get_old_act_probs()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.model.a_t, logits=self.model.get_act_logits())

        if self.off_policy:
            real_act_onehot = tf.one_hot(self.model.a_t, depth=act_nums)
            ro_t = tf.reduce_sum(
                real_act_onehot * act_probs, axis=-1
            ) / (tf.reduce_sum(
                real_act_onehot * self.old_act_probs,
                axis=-1) + 1e-8)
            ro_t = tf.clip_by_value(ro_t, 0.8, 1.2, "IS_clip")

            per_example_loss = per_example_loss * ro_t

        # this is the loss that corresponding to notations in self.apply_gradient
        self.a_loss = tf.reduce_mean(
            self.get_slots(
                self.build_icm_loss(
                    self.build_entropy_loss(
                        self.A_t * per_example_loss))))

    def _build_reset_op(self):
        super()._build_reset_op()

    def _build_train_op(self):
        grads_and_vars = self.compute_gradient(
            self.a_loss, self.model.get_actor_current_variables())
        self.a_current_train_op = self.apply_gradients(grads_and_vars)

    def update(self, _global_step, feed_dict):
        _, loss = self.sess.run([self.a_current_train_op, self.a_loss], feed_dict)
        return loss

    def act(self, feed_dict):
        return self.sess.run(
            [self.model.get_current_act(),
             self.model.get_current_act_probs(),
             self.model.get_current_act_logits()],
            feed_dict)
