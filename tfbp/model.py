import os
import tensorflow as tf
from tfbp.helpers.definitions import model_path, model_dir, model_name
from tfbp.dl.blocks import conv_block, lstm_block


class Model:
  path = model_path
  learning_rate = 0.0001

  def exists(self):
    return os.path.exists(model_dir) and len([f for f in os.listdir(model_dir) if f.startswith(model_name)]) > 0

  def build_network(self, batch_size=4, feed_previous=False):
    print 'Building network...'
    
    # Create placeholders for inputs
    # Construct blocks of NN using imported blocks (currently just conv and lstm)

    # Define loss (Ex:)
    loss = 0.0
    # for i in range(len(output_words)):
    #   loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_words[:, i, :],
    #                                                                  logits=output_words[i]))

    # Create minimize_loss function
    minimize_loss = None

    if not feed_previous:
      opt = tf.train.AdamOptimizer(self.learning_rate)
      minimize_loss = opt.minimize(loss)

    # return (inputs,), last_output, (loss, minimize_loss)