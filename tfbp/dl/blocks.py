import tensorflow as tf
from layers import conv_2d, fully_connected


def conv_block(x, num_filters=32, filter_dims=[5, 5], fc_size=1024, scope='conv_block', batch_size=4):
  with tf.variable_scope(scope):
    # Downsample image with stride [3, 3]
    a = conv_2d(x, dims=[7, 7], filters=num_filters, strides=[3, 3], std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv1')

    # NO downsampling with stride [1, 1]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[1, 1], std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv2')

    num_filters = 2 * num_filters
    # Downsample image with stride [2, 2]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[2, 2], std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv3')

    # NO downsampling with stride [1, 1]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[1, 1], std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv4')

    num_filters = 2 * num_filters
    
    # Downsample image with stride [2, 2]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[2, 2], std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv5')

    # NO downsampling with stride [1, 1]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[1, 1], std='xavier',
               padding='SAME', activation=tf.nn.relu, scope='conv6')

    num_filters = 32
    
    # Downsample image with stride [2, 2]
    a = conv_2d(a, filter_dims, filters=num_filters, strides=[2, 2], std='xavier',
               padding='VALID', activation=tf.nn.relu, scope='conv7')

    # Convert to vector with fully_connected layer
    a = tf.reshape(a, shape=[batch_size, -1])
    a = fully_connected(a, output_units=fc_size, activation=tf.nn.relu, std='xavier', scope='fc')

    print 'Output vector of conv_block is: {}'.format(a)
    
    return a


def lstm_block(x, v, t, lstm_size=512, vocab_size=52, num_words=30, feed_previous=False,
               scope='lstm_block', reuse=False, batch_size=4):
  with tf.variable_scope(scope, reuse=reuse):
    with tf.variable_scope('lstm_1', reuse=reuse):
      lstm_first = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=reuse)
      state_first = lstm_first.zero_state(batch_size, tf.float32)

      o_1, state_first = lstm_first(x[:, 0, :], state_first)

      r = tf.concat([o_1, v, t], axis=1)

    with tf.variable_scope('lstm_2', reuse=reuse):
      lstm_second = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=reuse)
      state_second = lstm_second.zero_state(batch_size, tf.float32)

      o_2, state_second = lstm_second(r, state_second)

    o = fully_connected(o_2, output_units=vocab_size, std='xavier', activation=tf.identity, reuse=False, scope='lstm_fc')

    if feed_previous:
      print o
      o = tf.nn.softmax(o)
      print o
      o = softmax_to_binary(o, axis=1)
      print o

  with tf.variable_scope(scope, reuse=True):
    # Teacher training - we feed in a list of words so we don't need to feed back in the output of the lstm.
    outputs = [o]
    
    for i in range(num_words - 1):
      if feed_previous:
        word = o
      else:
        word = x[:, i + 1, :]

      with tf.variable_scope('lstm_1', reuse=True):
        print word
        o, state_first = lstm_first(word, state_first)

      o = tf.concat([o, v, t], axis=1)

      with tf.variable_scope('lstm_2', reuse=True):
        o, state_second = lstm_second(o, state_second)

      o = fully_connected(o, output_units=vocab_size, std='xavier', activation=tf.identity, reuse=True, scope='lstm_fc')

      if feed_previous:
        o = tf.nn.softmax(o)
        o = softmax_to_binary(o, axis=1)
        outputs.append(o)
      else:
        outputs.append(o)

  return outputs


def softmax_to_binary(x, axis=0):
  shape = tf.shape(x)
  
  m = tf.reduce_max(x, axis=axis, keep_dims=True)
  s = tf.ones(shape=shape) * m
  b = tf.greater_equal(x, s)
  
  zeros = tf.zeros(shape=shape)
  ones = tf.ones(shape=shape)
  out = tf.where(b, ones, zeros)
  
  return out