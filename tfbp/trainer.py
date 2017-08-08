import json
import os
import numpy as np
import tensorflow as tf
from tfbp.helpers.definitions import global_step_path
from tfbp.model import Model
from tfbp.helpers import dataset


class Trainer:
  batch_size = 4
  print_every = 10
  save_every = 100
  train_steps = 10000
  
  def __init__(self):
    self.X_train, self.Y_train = dataset.train()
    self.X_val, self.Y_val = dataset.val()
    self.X_test, self.Y_test = dataset.test()
    
    self.model = Model()
    
    inputs, output, loss_info = self.model.build_network()
    
    # self.x_image, self.x_words, self.y_words, self.y_past = inputs
    # self.output_words = output
    
    self.loss, self.minimize_loss = loss_info
    
    # Create a new session and initialize globals
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    
    # Create our saver
    self.saver = tf.train.Saver(max_to_keep=200)
    
    # Restore prev model if exists
    if self.model.exists():
      self.saver.restore(self.sess, self.model.path)
    
    # Get stored global step value
    self.global_step = self.get_gstep()

  def get_batch(self):
    inds = list(np.random.choice(range(self.X_train.shape[0]), size=self.batch_size, replace=False))
    inds.sort()

    x = self.X_train[inds]
    y = self.Y_train[inds]
    
    # Use x and y to select random batches
    return

  def train(self):
    print 'Starting to train. Press Ctrl+C to save and exit.'
    
    try:
      for i in range(self.train_steps)[self.global_step:]:
        print '{}/{}'.format(i, self.train_steps)
  
        feed_dict = {}  # use return vals from get_batch() to populate feed_dict
        
        self.sess.run(self.minimize_loss, feed_dict)
  
        self.global_step += 1
  
        if not self.global_step % self.print_every:
          loss = self.sess.run(self.loss, feed_dict)
          print "Iteration {}: training loss = {}".format(i, loss)
  
        if not self.global_step % self.save_every:
          self.save_session()
          
    except (KeyboardInterrupt, SystemExit):
      print 'Interruption detected, exiting the program...'
    except BaseException, e:
      print 'Unexpected error during training: {}'.format(e)

    self.save_session()
  
  def save_session(self):
    print 'Saving session...'
    self.set_gstep()
    self.saver.save(self.sess, self.model.path)

  def get_gstep(self):
    if not os.path.exists(global_step_path):
      self.set_gstep()
      return 0
    
    with open(global_step_path) as f:
      return json.load(f).get('val') or 0

  def set_gstep(self):
    with open(global_step_path, 'w+') as f:
      f.write(json.dumps({'val': self.global_step or 0}, indent=2))
  
  def predict_test_set(self):
    # iterating over self.X_test, call single_predict()
    return
  
  def single_predict(self):
    prediction = np.zeros((1,))
    return prediction