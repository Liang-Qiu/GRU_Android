#copyright 2016 LiangKlausQiu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

import reader


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class Input(object):
  """The input data."""

  def __init__(self, is_testing, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    if is_testing:
      self.input_data = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="input")
      self.targets = tf.zeros([batch_size, num_steps], dtype=tf.int32)
#      self.data_len = tf.Variable(0);
#      for i in range(num_steps):
#	print("input_data[]:%d" %(self.input_data[0, i]))
#        if self.input_data[0, i] == 9999:          #<eos>
#	  break;
#	else: 
#	  self.data_len = tf.add(self.data_len, 1)
#      print("data_len: %d" %(self.data_len))
      print("set as placeholder")
     
    else:
      self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)
#      self.data_len = len(data);
      print("read from pipeline")
      
@property
def input_data(self):
  return self.input_data
#@property
#def data_len(self):
#  return self.data_len


class Model(object):
  """The GRU model."""

  def __init__(self, is_training, is_testing, config, input_):
    self._config = config
    self._input = input_
    ########################## self._config/self._input ############################
    self._input_data = input_data = input_.input_data
    batch_size = input_.batch_size
    num_steps = input_.num_steps
    targets = input_.targets
#    data_len = input_.data_len

    hidden_size = config.hidden_size
    num_layers = config.num_layers
    vocab_size = config.vocab_size
    keep_prob = config.keep_prob
    max_grad_norm = config.max_grad_norm

    # model definition
    gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)

    if is_training and keep_prob < 1:       
      gru_cell = tf.nn.rnn_cell.DropoutWrapper(
        gru_cell, output_keep_prob=keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * num_layers, state_is_tuple=True)
    self._initial_state = cell.zero_state(batch_size, tf.float32)
    ############################ self._initial_state ###############################

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, hidden_size], dtype=tf.float32)
      self._inputs = inputs = tf.nn.embedding_lookup(embedding, input_data)

    if is_training and keep_prob < 1:
      inputs = tf.nn.dropout(inputs, keep_prob)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):

  #  if not is_testing:
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
        self._final_state = state


 #     else:
  #      time_step1 = tf.constant(0)
#	while_condition1 = lambda time_step1: tf.less(time_step1, data_len)
#	def body1(time_step1):
 #         if tf.greater(time_step1, 0): tf.get_variable_scope().reuse_variables()
#	  (cell_output, state) = cell(inputs[:, time_step1, :], state)
#	  outputs.append(cell_output)
 #         self._final_state = state
#	  return [tf.add(time_step1, 1)]
#        r1 = tf.while_loop(while_condition1, body1, [time_step1])

#	time_step2 = tf.constant(num_steps-1)
#	while_condition2 = lambda time_step2: tf.greater_equal(time_step2, data_len)
#	def body2(time_step2):
#	  cell_output = tf.zeros([batch_size, hidden_size])
#	  outputs.append(cell_output)  
#	  return [tf.subtract(time_step2, 1)]
 #       r2 = tf.while_loop(while_condition2, body2, [time_step2]) 
  

#      if is_testing:
#        (outputs, self._final_state) = tf.nn.dynamic_rnn(cell, inputs, sequence_length=[data_len], initial_state=self._initial_state) 
#      else:
#        (outputs, self._final_state) = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state) 
        

#    self._final_state = state
    ############################### self._final_state ##############################
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
    self._output = output
    softmax_w = tf.get_variable(
        "softmax_w", [hidden_size, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    self._logits = logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name="logits")
    ############################### self._logits ###################################
    if is_testing:
      self._result = tf.slice(logits, [num_steps - 1, 0], [1, vocab_size], name="result")
      return

    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
	[tf.reshape(targets, [-1])],
	[tf.ones([batch_size * num_steps], dtype=tf.float32)])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    ############################## self._cost ######################################

    if not is_training:
    	return

    self._lr = tf.Variable(0.0, trainable=False)
    ############################# self._lr #########################################
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
	global_step=tf.contrib.framework.get_or_create_global_step())
    ############################ self._train_op ####################################
    
    self._new_lr = tf.placeholder(
    	tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data
  
  @property
  def inputs(self):
    return self._inputs

  @property
  def output(self):
    return self._output

  @property
  def input(self):
    return self._input
    
  @property
  def config(self):
    return self._config

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def final_state(self):
    return self._final_state

  @property
  def result(self):
    return self._result

  @property
  def logits(self):
    return self._logits

  @property
  def cost(self):
    return self._cost

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
