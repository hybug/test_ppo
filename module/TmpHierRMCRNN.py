# coding: utf-8

import collections

import numpy as np
import tensorflow as tf

from module import gelu

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export


def wrap_activations_get(name):
    if name == "gelu":
        return gelu
    return activations.get(name)


def wrap_activations_serialize(activation):
    try:
        return activations.serialize(activation)
    except:
        return "gelu"


class TmpHierRMCRNNCell(DropoutRNNCellMixin, Layer):
    def __init__(self,
                 time_scale,
                 size_per_head,
                 num_attention_heads,
                 num_memory_slots,
                 use_relative_position=True,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 mlp_activation='gelu',
                 forget_bias=1.0,
                 input_bias=0.0,
                 sigma_bias=-4.6,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='truncated_normal',
                 mlp_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 attention_regularizer=None,
                 mlp_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 attention_constraint=None,
                 mlp_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(TmpHierRMCRNNCell, self).__init__(**kwargs)
        self.time_scale = time_scale
        self.units = num_attention_heads * size_per_head
        self.num_memory_slots = num_memory_slots
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.use_relative_position = use_relative_position

        self.activation = wrap_activations_get(activation)
        self.recurrent_activation = wrap_activations_get(recurrent_activation)
        self.mlp_activation = wrap_activations_get(mlp_activation)
        self.forget_bias = forget_bias
        self.input_bias = input_bias
        self.sigma_bias = sigma_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_initializer = initializers.get(attention_initializer)
        self.mlp_initializer = initializers.get(mlp_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.attention_regularizer = regularizers.get(attention_regularizer)
        self.mlp_regularizer = regularizers.get(mlp_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.attention_constraint = constraints.get(attention_constraint)
        self.mlp_constraint = constraints.get(mlp_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
        # and fixed after 2.7.16. Converting the state_size to wrapper around
        # NoDependency(), so that the base_layer.__setattr__ will not convert it to
        # ListWrapper. Down the stream, self.states will be a list since it is
        # generated from nest.map_structure with list, and tuple(list) will work
        # properly.
        self.state_size = [
            1, 2 * (4 + self.num_memory_slots) * self.units]
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        with tf.variable_scope("prior"):
            self.p_ws = self._build_weights(input_dim=self.units)
            self.p_ws = self._build_weights(input_dim=2 * self.units)
        with tf.variable_scope("posterior"):
            self.q_ws = self._build_weights(input_dim=input_dim + 3 * self.units)
        if self.use_relative_position:
            self.rel_table = self._make_relative_position_embedding_table()
            self.pos_table = self._make_relative_position_table()
        self.built = True

    def _build_weights(self, input_dim):
        d = collections.OrderedDict()
        d["input_kernel"] = self.add_weight(
            shape=(input_dim, self.units),
            name='input_kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        d["gate_kernel"] = self.add_weight(
            shape=(input_dim, self.units * 2),
            name='gate_kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        d["recurrent_kernel"] = self.add_weight(
            shape=(self.units, self.units * 2),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        d["attention_kernel"] = self.add_weight(
            shape=(self.units, self.units * 3),
            name='attention_kernel',
            initializer=self.attention_initializer,
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint)
        d["mlp_kernel"] = self.add_weight(
            shape=(self.units, self.units * 2),
            name='mlp_kernel',
            initializer=self.mlp_initializer,
            regularizer=self.mlp_regularizer,
            constraint=self.mlp_constraint)
        d["input_bias"] = self.add_weight(
            shape=(self.units,),
            name='input_bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        d["recurrent_bias"] = self.add_weight(
            shape=(self.units * 2,),
            name='recurrent_bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        d["attention_bias"] = self.add_weight(
            shape=(self.units * 3,),
            name='attention_bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        d["mlp_bias"] = self.add_weight(
            shape=(self.units * 2,),
            name='mlp_bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        d["layer_norm_gamma"] = self.add_weight(
            shape=(1, 1, self.units * 2),
            name='layer_norm_gamma',
            initializer=self.kernel_initializer)
        d["layer_norm_beta"] = self.add_weight(
            shape=(self.units * 2,),
            name='layer_norm_beta',
            initializer=self.bias_initializer)

        ##########################
        d["random_kernel"] = self.add_weight(
            shape=(self.units, self.units * 2),
            name='random_kernel',
            initializer=self.mlp_initializer,
            regularizer=self.mlp_regularizer,
            constraint=self.mlp_constraint)
        d["random_bias"] = self.add_weight(
            shape=(self.units * 2,),
            name='random_bias',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        ##########################
        if self.use_relative_position:
            d["rel_kernel"] = self.add_weight(
                shape=(self.units, self.units),
                name='rel_kernel',
                initializer=self.attention_initializer,
                regularizer=self.attention_regularizer,
                constraint=self.attention_constraint)
        return d

    def _make_relative_position_embedding_table(self):
        table = []
        for p in range(-self.num_memory_slots, self.num_memory_slots + 1):
            row = []
            for i in range(self.units):
                if i % 2 == 0:
                    row.append(np.sin(p / np.power(1e+4, i / self.units)))
                else:
                    row.append(np.cos(p / np.power(1e+4, (i - 1) / self.units)))
            table.append(row)
        return K.constant(table)

    def _make_relative_position_table(self):
        from_position = np.arange(self.num_memory_slots + 1)[:, None]
        to_position = np.arange(self.num_memory_slots + 1)[None, :]
        table = from_position - to_position + self.num_memory_slots
        return K.cast(K.one_hot(table, 2 * self.num_memory_slots + 1), tf.float32)

    def _attention_layer(self, memory_plus_inputs, ws):
        from_length = self.num_memory_slots + 1
        to_length = self.num_memory_slots + 1

        q_bias, k_bias, v_bias = array_ops.split(ws["attention_bias"], 3, axis=0)

        # [B, F, N, H]
        query_layer = K.dot(
            memory_plus_inputs, ws["attention_kernel"][:, :self.units])
        query_layer = K.bias_add(query_layer, q_bias)
        query_layer = array_ops.reshape(
            query_layer,
            [-1, from_length, self.num_attention_heads, self.size_per_head])
        # [B, N, F, H]
        query_layer1 = array_ops.transpose(query_layer, perm=[0, 2, 1, 3])
        # [B*N, F, H]
        query_layer = array_ops.reshape(
            query_layer1, shape=[-1, from_length, self.size_per_head])

        # [B, T, N, H]
        key_layer = K.dot(
            memory_plus_inputs, ws["attention_kernel"][:, self.units:self.units * 2])
        key_layer = K.bias_add(key_layer, k_bias)
        key_layer = array_ops.reshape(
            key_layer,
            [-1, to_length, self.num_attention_heads, self.size_per_head])
        # [B, N, T, H]
        key_layer = array_ops.transpose(key_layer, perm=[0, 2, 1, 3])
        # [B*N, T, H]
        key_layer = array_ops.reshape(
            key_layer, shape=[-1, to_length, self.size_per_head])

        # [B, T, N, H]
        value_layer = K.dot(
            memory_plus_inputs, ws["attention_kernel"][:, self.units * 2:self.units * 3])
        value_layer = K.bias_add(value_layer, v_bias)
        value_layer = array_ops.reshape(
            value_layer,
            [-1, to_length, self.num_attention_heads, self.size_per_head])
        # [B, N, T, H]
        value_layer = array_ops.transpose(value_layer, perm=[0, 2, 1, 3])
        # [B*N, T, H]
        value_layer = array_ops.reshape(
            value_layer, shape=[-1, to_length, self.size_per_head])

        # [B*N, F, T]
        attention_scores = K.batch_dot(query_layer, key_layer, axes=[2, 2])

        if self.use_relative_position:
            # [F+T-1, N*H]
            r = K.dot(self.rel_table, ws["rel_kernel"])
            # [F+T-1, N, H]
            r = array_ops.reshape(
                r, [-1, self.num_attention_heads, self.size_per_head])
            # [B, N, F, F+T-1]
            bd = tf.einsum("bnfh,lnh->bnfl", query_layer1, r)
            # [B*N, F, F+T-1]
            bd = array_ops.reshape(
                bd, [-1, from_length, from_length + to_length - 1])
            # [B*N, F, T]
            bd = tf.einsum("bfl,ftl->bft", bd, self.pos_table)
            # [B*N, F, T]
            attention_scores += bd

        # [B*N, F, T]
        attention_scores = attention_scores / K.cast(self.size_per_head, tf.float32)

        # [B*N, F, T]
        attention_probs = K.softmax(attention_scores)

        # [B*N, F, H]
        context_layer = K.batch_dot(attention_probs, value_layer, axes=[2, 1])

        # [B, N, F, H]
        context_layer = array_ops.reshape(
            context_layer,
            [-1, self.num_attention_heads, from_length, self.size_per_head])

        # [B, F, N, H]
        context_layer = array_ops.transpose(context_layer, perm=[0, 2, 1, 3])

        # [B, F, N*H]
        context_layer = array_ops.reshape(
            context_layer,
            [-1, from_length, self.num_attention_heads * self.size_per_head])

        return context_layer

    def _attend_over_memory(self, inputs, memory, ws):
        inputs = K.dot(inputs, ws["input_kernel"])
        inputs = K.bias_add(inputs, ws["input_bias"])
        inputs = K.expand_dims(inputs, axis=1)

        memory_plus_inputs = K.concatenate([memory, inputs], axis=1)
        context_layer = self._attention_layer(memory_plus_inputs, ws)

        beta1, beta2 = array_ops.split(ws["layer_norm_beta"], 2, axis=0)
        mlp_b1, mlp_b2 = array_ops.split(ws["mlp_bias"], 2, axis=0)

        context_layer = memory_plus_inputs + context_layer
        context_layer = K.l2_normalize(
            context_layer - K.mean(context_layer, axis=-1, keepdims=True),
            axis=-1)
        context_layer = context_layer * ws["layer_norm_gamma"][:, :, :self.units]
        context_layer = K.bias_add(context_layer, beta1)

        mlp_layer = K.dot(context_layer, ws["mlp_kernel"][:, :self.units])
        mlp_layer = K.bias_add(mlp_layer, mlp_b1)
        mlp_layer = self.mlp_activation(mlp_layer)
        mlp_layer = K.dot(mlp_layer, ws["mlp_kernel"][:, self.units:])
        mlp_layer = K.bias_add(mlp_layer, mlp_b2)

        context_layer = context_layer + mlp_layer
        context_layer = K.l2_normalize(
            context_layer - K.mean(context_layer, axis=-1, keepdims=True),
            axis=-1)
        context_layer = context_layer * ws["layer_norm_gamma"][:, :, self.units:]
        context_layer = K.bias_add(context_layer, beta2)

        new_memory, outputs = array_ops.split(
            context_layer, [self.num_memory_slots, 1], axis=1)
        outputs = K.squeeze(outputs, axis=1)

        return outputs, new_memory

    def _input_and_forget_gates(self, inputs, memory, ws):
        gates = K.expand_dims(
            K.dot(inputs, ws["gate_kernel"]), axis=1
        ) + K.dot(memory, ws["recurrent_kernel"])
        gates = K.bias_add(gates, ws["recurrent_bias"])

        input_gate, forget_gate = array_ops.split(
            gates, num_or_size_splits=2, axis=2)

        input_gate = self.recurrent_activation(input_gate + self.input_bias)
        forget_gate = self.recurrent_activation(forget_gate + self.forget_bias)

        return input_gate, forget_gate

    def _call_one_layer(self, inputs, flatten_memory, training, ws):
        dp_mask = self.get_dropout_mask_for_cell(
            inputs, training, count=1)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            flatten_memory, training, count=1)

        if 0 < self.dropout < 1:
            inputs = inputs * dp_mask[0]
        if 0 < self.recurrent_dropout < 1:
            flatten_memory = flatten_memory * rec_dp_mask[0]

        memory = array_ops.reshape(
            flatten_memory, shape=[-1, self.num_memory_slots, self.units])

        input_gate, forget_gate = self._input_and_forget_gates(inputs, memory, ws)
        hs, new_memory = self._attend_over_memory(inputs, memory, ws)

        next_memory = input_gate * new_memory + forget_gate * memory

        flatten_next_memory = array_ops.reshape(
            next_memory, shape=[-1, self.num_memory_slots * self.units])

        mus_and_log_sigmas = K.dot(hs, ws["random_kernel"])
        mus_and_log_sigmas = K.bias_add(mus_and_log_sigmas, ws["random_bias"])
        mus, log_sigmas = array_ops.split(mus_and_log_sigmas, 2, axis=-1)
        sigmas = K.log(1.0 + K.exp(log_sigmas + self.sigma_bias))
        zs = K.random_normal(shape=K.shape(mus)) * sigmas + mus

        return zs, mus, sigmas, hs, flatten_next_memory

    def call(self, inputs, states, training=None):
        count, state_in = states
        (p_zs, p_mus, p_sigmas, p_hs, p_flatten_memory,
         q_zs, q_mus, q_sigmas, q_hs, q_flatten_memory
         ) = array_ops.split(
            state_in,
            [self.units, self.units, self.units, self.units,
             self.units * self.num_memory_slots,
             self.units, self.units, self.units, self.units,
             self.units * self.num_memory_slots],
            axis=-1)

        # prior_inputs = q_hs
        prior_inputs = K.concatenate([q_hs, p_hs])
        (p_next_zs, p_next_mus, p_next_sigmas,
         p_next_hs, p_next_flatten_memory) = self._call_one_layer(
            prior_inputs, p_flatten_memory, training, self.p_ws)

        p_next_zs = tf.where_v2(
            tf.floormod(count, self.time_scale) > 0,
            p_zs, p_next_zs)
        p_next_mus = tf.where_v2(
            tf.floormod(count, self.time_scale) > 0,
            p_mus, p_next_mus)
        p_next_sigmas = tf.where_v2(
            tf.floormod(count, self.time_scale) > 0,
            p_sigmas, p_next_sigmas)
        p_next_hs = tf.where_v2(
            tf.floormod(count, self.time_scale) > 0,
            p_hs, p_next_hs)
        p_next_flatten_memory = tf.where_v2(
            tf.floormod(count, self.time_scale) > 0,
            p_flatten_memory, p_next_flatten_memory)

        posterior_inputs = K.concatenate(
            [inputs, p_next_mus, p_next_sigmas, q_zs])
        (q_next_zs, q_next_mus, q_next_sigmas,
         q_next_hs, q_next_flatten_memory) = self._call_one_layer(
            posterior_inputs, q_flatten_memory, training, self.q_ws)

        state_out = K.concatenate(
            [p_next_zs, p_next_mus, p_next_sigmas,
             p_next_hs, p_next_flatten_memory,
             q_next_zs, q_next_mus, q_next_sigmas,
             q_next_hs, q_next_flatten_memory])

        return ({"p_zs": p_next_zs,
                 "p_mus": p_next_mus,
                 "p_sigmas": p_next_sigmas,
                 "q_zs": q_next_zs,
                 "q_mus": q_next_mus,
                 "q_sigmas": q_next_sigmas},
                [count + 1.0, state_out])

    def get_config(self):
        config = {
            'units':
                self.units,
            'num_memory_slots':
                self.num_memory_slots,
            'num_attention_heads':
                self.num_attention_heads,
            'activation':
                wrap_activations_serialize(self.activation),
            'recurrent_activation':
                wrap_activations_serialize(self.recurrent_activation),
            'mlp_activation':
                wrap_activations_serialize(self.mlp_activation),
            'forget_bias':
                self.forget_bias,
            'input_bias':
                self.input_bias,
            'sigma_bias':
                self.sigma_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'attention_initializer':
                initializers.serialize(self.attention_initializer),
            'mlp_initializer':
                initializers.serialize(self.mlp_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'attention_regularizer':
                regularizers.serialize(self.attention_regularizer),
            'mlp_regularizer':
                regularizers.serialize(self.mlp_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'attention_constraint':
                constraints.serialize(self.attention_constraint),
            'mlp_constraint':
                constraints.serialize(self.mlp_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(TmpHierRMCRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))


class TmpHierRMCRNN(RNN):
    def __init__(self,
                 time_scale,
                 size_per_head,
                 num_attention_heads,
                 num_memory_slots,
                 use_relative_position=True,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 mlp_activation='relu',
                 forget_bias=1.0,
                 input_bias=0.0,
                 sigma_bias=-4.6,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='truncated_normal',
                 mlp_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 attention_regularizer=None,
                 mlp_regularizer=None,
                 activity_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 attention_constraint=None,
                 mlp_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = TmpHierRMCRNNCell(
            time_scale=time_scale,
            size_per_head=size_per_head,
            num_memory_slots=num_memory_slots,
            num_attention_heads=num_attention_heads,
            use_relative_position=use_relative_position,
            activation=activation,
            recurrent_activation=recurrent_activation,
            mlp_activation=mlp_activation,
            forget_bias=forget_bias,
            input_bias=input_bias,
            sigma_bias=sigma_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            attention_initializer=attention_initializer,
            mlp_initializer=mlp_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            attention_regularizer=attention_regularizer,
            mlp_regularizer=mlp_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            attention_constraint=attention_constraint,
            mlp_constraint=mlp_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(TmpHierRMCRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        return super(TmpHierRMCRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {
            'units':
                self.units,
            'num_memory_slots':
                self.num_memory_slots,
            'num_attention_heads':
                self.num_attention_heads,
            'activation':
                wrap_activations_serialize(self.activation),
            'recurrent_activation':
                wrap_activations_serialize(self.recurrent_activation),
            'mlp_activation':
                wrap_activations_serialize(self.mlp_activation),
            'forget_bias':
                self.forget_bias,
            'input_bias':
                self.input_bias,
            'sigma_bias':
                self.sigma_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'attention_initializer':
                initializers.serialize(self.attention_initializer),
            'mlp_initializer':
                initializers.serialize(self.mlp_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'attention_regularizer':
                regularizers.serialize(self.attention_regularizer),
            'mlp_regularizer':
                regularizers.serialize(self.mlp_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'attention_constraint':
                constraints.serialize(self.attention_constraint),
            'mlp_constraint':
                constraints.serialize(self.mlp_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(TmpHierRMCRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)
