# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict
import re

from utils import get_shape


class base(object):
    def __init__(self,
                 sess,
                 model,
                 optimizer,
                 gamma=1.0,
                 entropy=0.01,
                 trace_decay_rate=None,
                 icm=None,
                 grad_norm_clip=None,
                 scope=None):
        """
        sess:
            the tf session that runs the algorithm

        optimizer:
            the optimizer used to update the gradient,
            for instance, tf.train.AdamOptimizer()
            should have
                .compute_gradients
                .apply_gradients

        gamma: discount rate
            this is not used for calculating return within this class,
            but when updating trace for episodic tasks, gamma is necessary

        trace_decay_rate:
            whether to use eligibility trace when updating,
            the formula is identical to Sutton's 2nd edition chapter 12&13
            If None, no trace is used, else, using trace with the given rate.
            trace decay rate should be in [1.1, 1.1]

        entropy:
            if not None, add entropy of act distribution to loss,
            loss <- loss - alpha * H(act_dist)

        icm:
            whether to use Intrinsic Curiosity Module
            </Curiosity-driven Exploration by Self-supervised Prediction/>

        grad_norm_clip:
            whether to clip_global_norm(grads)

        """
        with tf.variable_scope(scope):
            self.sess = sess
            self.model = model
            self.optimizer = optimizer
            self.gamma = gamma
            self.entropy = entropy
            self.trace_decay_rate = trace_decay_rate
            self.icm = icm
            self.grad_norm_clip = grad_norm_clip
            self.build_init_target_op()
            self.build_trace()

            self._build_loss_fn()
            self._build_reset_op()
            self._build_train_op()

    def _build_loss_fn(self):
        self.loss = 0.0

    def _build_reset_op(self):
        self.reset_op = []
        if self.trace_decay_rate is not None:
            for z_var in self.z.values():
                self.reset_op.append(tf.assign(z_var, tf.zeros_like(z_var)))
            self.reset_op.append(tf.assign(self.I, 1.0))

    def _build_train_op(self):
        self.train_op = []

    def apply_gradients(self, grads_and_vars):
        """
        please carefully notice that the trace version here is
        a little bit different from that on Sutton's 2nd ed page 275.
        As delta term in the book works only when updating variables,
        and trace only follows the previous states.
        In our implementation, delta is moved to the previous step
        when updating traces, this is for the concern of efficiency
        when training in a mini-batch, otherwise we need to construct
        a mini-batch of traces to follow the traces.
        """
        op = []
        if self.trace_decay_rate is not None:
            new_grads_and_vars = []
            for grad, var in grads_and_vars:
                z_var = self.z[var.name]
                op.append(
                    tf.assign(
                        z_var,
                        self.gamma * self.trace_decay_rate * z_var + self.I * grad))
                new_grads_and_vars.append((z_var, var))
            op.append(self.optimizer.apply_gradients(new_grads_and_vars))
            op.append(tf.assign(self.I, self.I * self.gamma))
        else:
            op = self.optimizer.apply_gradients(grads_and_vars)
        return op

    def compute_gradient(self, loss, var_list, summary=False):
        if loss.__class__ == list:
            loss = loss[0]
        grads_and_vars = self.optimizer.compute_gradients(
            loss, var_list=var_list)
        if self.grad_norm_clip is not None:
            grads, _ = tf.clip_by_global_norm(
                [t[0] for t in grads_and_vars],
                self.grad_norm_clip)

            if summary:
                tf.summary.scalar("global_norm", _)

            grads_and_vars = list(
                zip(grads, [t[1] for t in grads_and_vars]))
        return grads_and_vars

    def reset(self):
        self.sess.run(self.reset_op)

    def build_trace(self):
        if self.trace_decay_rate is not None:

            get_op = getattr(self.model, "get_variables", None)
            assert get_op is not None
            assert callable(get_op)

            self.I = tf.get_variable(name="cumulated_discount_rate",
                                     shape=[],
                                     dtype=tf.float32,
                                     initializer=tf.ones_initializer(),
                                     trainable=False)
            self.z = OrderedDict()
            with tf.variable_scope("eligibility_trace"):
                for var in self.model.get_variables():
                    var_name = var.name
                    m = re.match("^(.*):\\d+$", var_name)
                    if m is not None:
                        var_name = m.group(1)
                    z_var = tf.get_variable(
                        name=var_name,
                        shape=self.get_shape(var),
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                        trainable=False)
                    self.z[var.name] = z_var

    def build_entropy_loss(self, loss):
        if self.entropy is not None:
            try:
                probs = self.model.get_current_act_probs()
                logits = self.model.get_current_act_logits()
                entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=probs, logits=logits, axis=-1)
                ent_alpha = self.entropy * entropy
                tf.summary.scalar("entropy", tf.reduce_mean(entropy))
                tf.summary.scalar("entropy_alpha", tf.reduce_mean(ent_alpha))
                if loss.__class__ == list:
                    loss = loss[0] - ent_alpha
                else:
                    loss = loss - ent_alpha
            except:
                print("WARNING: CANNOT MAKE ENTROPY LOSS")
        return loss

    def build_icm_loss(self, loss):
        if self.icm is not None:
            icm_loss, forward_loss, inverse_loss = self.model.get_icm_loss()
            tf.summary.scalar("ICM_loss", icm_loss)
            tf.summary.scalar("ICM_F_loss", forward_loss)
            tf.summary.scalar("ICM_I_loss", inverse_loss)
            if loss.__class__ == list:
                loss = loss[0] + self.icm * icm_loss
            else:
                loss = loss + self.icm * icm_loss
        return loss

    def assert_variables(self,
                         target_vars,
                         current_vars,
                         target_vars_to_current_vars):
        all_vars = target_vars + current_vars

        assert set(target_vars).union(set(current_vars)) == set(all_vars)

        target_scope = [var.name.split("/")[0] for var in target_vars]

        current_scope = [var.name.split("/")[0] for var in current_vars]

        var_list = OrderedDict()
        for var in all_vars:
            fresh_name = "/".join(var.name.split("/")[1:])
            if fresh_name not in var_list:
                var_list[fresh_name] = [var]
            else:
                var_list[fresh_name] += [var]

        for var_name, two_vars in var_list.items():
            t_var, c_var = two_vars
            if t_var.name.split("/")[0] in target_scope:
                assert c_var.name.split("/")[0] in current_scope
            else:
                t_var, c_var = c_var, t_var
                assert c_var.name.split("/")[0] in current_scope
                assert t_var.name.split("/")[0] in target_scope
            target_vars_to_current_vars[t_var] = c_var

    def build_init_target_op(self):
        init_target_op = getattr(self, "init_target_op", None)
        if init_target_op is None:
            self.init_target_op = []

    def get_slots(self, tensor):
        slots = getattr(self.model, "slots", None)
        if slots is not None:
            return tensor * tf.cast(slots, tf.float32)
        return tensor

    @staticmethod
    def get_shape(tensor):
        return get_shape(tensor)
