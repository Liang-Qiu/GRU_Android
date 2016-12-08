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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time 
import numpy as np
import tensorflow as tf

import reader
import GRU
from tensorflow.python.framework.graph_util import convert_variables_to_constants
#import freeze_graph

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string(
    "data_path", "/home/liangqiu/Documents/simple-examples/data", 
    "Where the training/test data is stored.")
flags.DEFINE_string(
    "save_path", "/home/liangqiu/Documents/GRU_output", 
    "Model output directory.")
#flags.DEFINE_bool(
#    "use_fp16", False,
#    "Train using 16-bit floats instead of 32bit floats.")

FLAGS = flags.FLAGS


#def data_type():
#  return tf.float if FLAGS.use_fp16 else tf.float32


def get_config():
  if FLAGS.model == "small":
    return GRU.SmallConfig()
  elif FLAGS.model == "medium":
    return GRU.MediumConfig()
  elif FLAGS.model == "large":
    return GRU.LargeConfig()
  elif FLAGS.model == "test":
    return GRU.TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def run_training(session, model):
  # train the model on the given data.
  start_time = time.time()
  costs = 0.0
  iters = 0

  fetches = {
      "cost": model.cost,
      "train_op": model.train_op
  }

  for step in range(model.input.epoch_size):
    vals = session.run(fetches)
    cost = vals["cost"]

    costs += cost
    iters += model.input.num_steps

  #  if step % (model.input.epoch_size // 10) == 0:
  #    print("%.2f perplexity: %.3f speed: %.0f wps" % 
  #          (step * 1.0 /model.input.epoch_size, np.exp(costs / iters),
  #	    iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def run_validation(session, model):
  # train the model on the given data.
  start_time = time.time()
  costs = 0.0
  iters = 0
   
  for step in range(model.input.epoch_size):
    cost = session.run(model.cost)

    costs += cost
    iters += model.input.num_steps
	   
  return np.exp(costs / iters)


def run_test(session, model, id_to_word=None, test_data=None):
  # test the model on the given data.
  start_time = time.time()
#  costs = 0.0
#  iters = 0

  fetches = {
      "result": model.result,
#      "cost": model.cost
  }
  test_feed = np.array([[1,3,3,4,5,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]], dtype=np.int32)		
  print(test_feed.shape[0])
  print(test_feed.shape[1])
  print(test_feed.dtype)
  test_dict = {
      model.input.input_data: test_feed
  }   
#  for step in range(model.input.epoch_size):
  print("input_data")
  print(session.run(model.input.input_data, feed_dict = test_dict))
#  print("data_len")
#  print(session.run(model.input.data_len, feed_dict = test_dict))
  print("inputs")
  print(session.run(model.inputs, feed_dict = test_dict))
  print("output")
  print(session.run(model.output, feed_dict = test_dict))
  vals = session.run(fetches, feed_dict=test_dict)
#  cost = vals["cost"]
  result = vals["result"]

#  costs += cost
#  iters += model.input.num_steps
	      
#  print("data_length: %s perplexity: %.3f speed: %.0f wps" % (len(test_data), np.exp(costs / iters), iters * model.input.batch_size / (time.time() - start_time)))
  
  print("%.8f" %(result[0, np.argmax(result)]))
  print(result)
  print("predicted word: %s" % (id_to_word[np.argmax(result)]))		   
#  return np.exp(costs / iters)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _, id_to_word = raw_data
 
  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
#  eval_config.num_steps = 4

 # checkpoint_prefix = os.path.join(FLAGS.save_path, "saved_checkpoint")
 # checkpoint_state_name = "checkpoint_state"
 # input_graph_name = "input_graph.pb"
 # output_graph_name = "output_graph.pb"

  g1 = tf.Graph()
  with g1.as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):
      train_input = GRU.Input(is_testing=False, config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse = None, initializer=initializer):
        mtrain = GRU.Model(is_training=True, is_testing=False, config=config, input_=train_input)

    with tf.name_scope("Valid"):
      valid_input = GRU.Input(is_testing=False, config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse = True, initializer=initializer):
        mvalid = GRU.Model(is_training=False, is_testing=False, config=config, input_=valid_input)

    with tf.name_scope("Test"):
      test_input = GRU.Input(is_testing=True, config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse = True, initializer=initializer):
        mtest = GRU.Model(is_training=False, is_testing=True, config=eval_config, input_=test_input)


    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
#    with tf.Session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
	mtrain.assign_lr(session, config.learning_rate * lr_decay)
        # train
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
	train_perplexity = run_training(session, mtrain)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        # valid
	valid_perplexity = run_validation(session,mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      run_test(session, mtest, id_to_word=id_to_word, test_data=test_data)
#      print("Test Perplexity: %.3f" % test_perplexity)
      

      # for FPGA use
     # Embedding = session.run(tf.get_default_graph().get_tensor_by_name("Model/embedding:0"))
      
     # Cell0_W1 = session.run(tf.get_default_graph().get_tensor_by_name("Model/RNN/MultiRNNCell/Cell0/GRUCell/Gates/Linear/Matrix:0"))
     # Cell0_B1 = session.run(tf.get_default_graph().get_tensor_by_name("Model/RNN/MultiRNNCell/Cell0/GRUCell/Gates/Linear/Bias:0"))
     # Cell0_W2 = session.run(tf.get_default_graph().get_tensor_by_name("Model/RNN/MultiRNNCell/Cell0/GRUCell/Candidate/Linear/Matrix:0"))
     # Cell0_B2 = session.run(tf.get_default_graph().get_tensor_by_name("Model/RNN/MultiRNNCell/Cell0/GRUCell/Candidate/Linear/Bias:0"))
      
     # Cell1_W1 = session.run(tf.get_default_graph().get_tensor_by_name("Model/RNN/MultiRNNCell/Cell1/GRUCell/Gates/Linear/Matrix:0"))
     # Cell1_B1 = session.run(tf.get_default_graph().get_tensor_by_name("Model/RNN/MultiRNNCell/Cell1/GRUCell/Gates/Linear/Bias:0"))
     # Cell1_W2 = session.run(tf.get_default_graph().get_tensor_by_name("Model/RNN/MultiRNNCell/Cell1/GRUCell/Candidate/Linear/Matrix:0"))
     # Cell1_B2 = session.run(tf.get_default_graph().get_tensor_by_name("Model/RNN/MultiRNNCell/Cell1/GRUCell/Candidate/Linear/Bias:0"))
     # softmax_W = session.run(tf.get_default_graph().get_tensor_by_name("Model/softmax_w:0"))
     # softmax_B = session.run(tf.get_default_graph().get_tensor_by_name("Model/softmax_b:0"))


     # para_file = open(FLAGS.save_path+"para_file.txt", "w")
     
     # para_file.write("# Embedding ")
     # np.savetxt(para_file, Embedding, fmt='%0.8f',delimiter=',')
     # para_file.write("# Cell0_W1 ")
     # np.savetxt(para_file, Cell0_W1, fmt='%0.8f',delimiter=',')
     # para_file.write("# Cell0_B1 ")
     # np.savetxt(para_file, Cell0_B1, fmt='%0.8f',delimiter=',')
     # para_file.write("# Cell0_W2 ")
     # np.savetxt(para_file, Cell0_W2, fmt='%0.8f',delimiter=',')
     # para_file.write("# Cell0_B2 ")
     # np.savetxt(para_file, Cell0_B2, fmt='%0.8f',delimiter=',')
     # para_file.write("# Cell1_W1 ")
     # np.savetxt(para_file, Cell1_W1, fmt='%0.8f',delimiter=',')
     # para_file.write("# Cell1_B1 ")
     # np.savetxt(para_file, Cell1_B1, fmt='%0.8f',delimiter=',')
     # para_file.write("# Cell1_W2 ")
     # np.savetxt(para_file, Cell1_W2, fmt='%0.8f',delimiter=',')
     # para_file.write("# Cell1_B2 ")
     # np.savetxt(para_file, Cell1_B2, fmt='%0.8f',delimiter=',')
     # para_file.write("# softmax_W ")
     # np.savetxt(para_file, softmax_W, fmt='%0.8f',delimiter=',')
     # para_file.write("# softmax_B ")
     # np.savetxt(para_file, softmax_B, fmt='%0.8f',delimiter=',')

     # para_file.close()

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
#	sv.saver.save(session, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)
	#      tf.train.write_graph(session.graph.as_graph_def(), FLAGS.save_path, input_graph_name)
#	input_graph_path = os.path.join(FLAGS.save_path, input_graph_name)
#	input_saver_def_path = ""
#	input_binary = False
#	input_checkpoint_path = checkpoint_prefix + "-0"
#	output_node_names = "Test/Model/logits"
#	restore_op_name = "save/restore_all"
#	filename_tensor_name = "save/Const:0"
#        output_graph_path = os.path.join(FLAGS.save_path, output_graph_name)
#	clear_devices = False

#	freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
#	                          input_binary, input_checkpoint_path,
#				  output_node_names, restore_op_name,
#				  filename_tensor_name, output_graph_path,
#				  clear_devices, None)
#        for v in tf.trainable_variables():
#	    vc = tf.constant(v.eval())
#	    tf.assign(v, vc, "assign_variables")
#        gru_graph = graph_util.extract_sub_graph(graph_def, ["Test/Model/logits"])
	
        gru_graph = convert_variables_to_constants(session, session.graph_def, ["Test/Model/result"])
	tf.train.write_graph(gru_graph, FLAGS.save_path, 'gru_graph.pb', as_text=False)
	tf.train.write_graph(gru_graph, FLAGS.save_path, "gru_graph.pbtxt", as_text=True)
        print("write gru_graph")

#  g2 = tf.Graph()
#  gru_input = {"Test/TestInput/raw_data": tf.placeholder(tf.int32, shape=20)}
#  with g2.as_default():
#    with tf.Session(graph=g2) as session:
#      tf.import_graph_def(tmp_graph, return_elements=["Test/Model/result"], name="")
#      tf.train.write_graph(session.graph_def, FLAGS.save_path, 'gru_graph.pbtxt', as_text=True)

if __name__ == "__main__":
  tf.app.run()
