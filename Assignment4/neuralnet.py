#! /usr/bin/env python
"""!
-----------------------------------------------------------------------------
File Name : neuralnet.py

Purpose:

Created: 01-Jun-2016 17:01:05 AEST
-----------------------------------------------------------------------------
Revision History



-----------------------------------------------------------------------------
S.D.G
"""
__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '01-Jun-2016 17:01:05 AEST'
__license__ = 'MPL v2.0'

# LICENSE DETAILS############################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# IMPORTS#####################################################################

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

##############################################################################

# In[2]:

pickle_file = '../Assignment1/notMNIST.pickle'

batch_size =128 
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory

# Reshape the image data for a neural net
train_dataset = train_dataset.reshape((train_dataset.shape[0],
    train_dataset.shape[1] * train_dataset.shape[2]))
train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
valid_dataset = valid_dataset.reshape((valid_dataset.shape[0],
    valid_dataset.shape[1] * valid_dataset.shape[2]))
valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
test_dataset = test_dataset.reshape((test_dataset.shape[0],
    test_dataset.shape[1] * test_dataset.shape[2]))
test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

graph = tf.Graph()

num_hidden = 1024 
num_hidden2 = 120
beta_reg = .001

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)

    layer1_weights = tf.Variable(tf.truncated_normal([
        image_size * image_size, num_hidden]))
    layer1_bias = tf.Variable(tf.truncated_normal([num_hidden]))
    layer2_weights = tf.Variable(tf.truncated_normal([
        num_hidden, num_labels]))
    layer2_bias = tf.Variable(tf.truncated_normal([num_labels])) 

    def model(data):
        layer1 = tf.nn.relu(tf.matmul(data, layer1_weights) + layer1_bias)
        layer2 = tf.matmul(layer1, layer2_weights) + layer2_bias
        return layer2 

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) +
        (beta_reg * tf.nn.l2_loss(layer1_weights)) + 
        (beta_reg * tf.nn.l2_loss(layer2_weights))
    )

    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.5, global_step, 100000, 0.6)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
          global_step=global_step)
    #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    #optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
