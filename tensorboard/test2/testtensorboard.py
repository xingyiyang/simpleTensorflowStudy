# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:20:03 2017

@author: xing
"""

# -*- coding=utf-8 -*-
# @author: 陈水平
# @date: 2017-02-09
# @description: implement a softmax regression model upon MNIST handwritten digits
# @ref: http://yann.lecun.com/exdb/mnist/

import gzip
import struct
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import tensorflow as tf

# MNIST data is stored in binary format, 
# and we transform them into numpy ndarray objects by the following two utility functions
def read_image(file_name):
    with gzip.open(file_name, 'rb') as f:
        buf = f.read()
        index = 0
        magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
        index += struct.calcsize('>IIII')

        image_size = '>' + str(images*rows*columns) + 'B'
        ims = struct.unpack_from(image_size, buf, index)
        
        im_array = np.array(ims).reshape(images, rows, columns)
        return im_array

def read_label(file_name):
    with gzip.open(file_name, 'rb') as f:
        buf = f.read()
        index = 0
        magic, labels = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')
        
        label_size = '>' + str(labels) + 'B'
        labels = struct.unpack_from(label_size, buf, index)

        label_array = np.array(labels)
        return label_array

print ("Start processing MNIST handwritten digits data...")
train_x_data = read_image("MNIST_data/train-images-idx3-ubyte.gz")
train_x_data = train_x_data.reshape(train_x_data.shape[0], -1).astype(np.float32)
train_y_data = read_label("MNIST_data/train-labels-idx1-ubyte.gz")
test_x_data = read_image("MNIST_data/t10k-images-idx3-ubyte.gz")
test_x_data = test_x_data.reshape(test_x_data.shape[0], -1).astype(np.float32)
test_y_data = read_label("MNIST_data/t10k-labels-idx1-ubyte.gz")

train_x_minmax = train_x_data / 255.0
test_x_minmax = test_x_data / 255.0

# Of course you can also use the utility function to read in MNIST provided by tensorflow
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
# train_x_minmax = mnist.train.images
# train_y_data = mnist.train.labels
# test_x_minmax = mnist.test.images
# test_y_data = mnist.test.labels

# We evaluate the softmax regression model by sklearn first
eval_sklearn = False
if eval_sklearn:
    print ("Start evaluating softmax regression model by sklearn...")
    reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    reg.fit(train_x_minmax, train_y_data)
    np.savetxt('coef_softmax_sklearn.txt', reg.coef_, fmt='%.6f')  # Save coefficients to a text file
    test_y_predict = reg.predict(test_x_minmax)
    print ("Accuracy of test set: %f" % accuracy_score(test_y_data, test_y_predict))

eval_tensorflow = True
batch_gradient = False

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
if eval_tensorflow:
    print ("Start evaluating softmax regression model by tensorflow...")
    # reformat y into one-hot encoding style
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_y_data)
    train_y_data_trans = lb.transform(train_y_data)
    test_y_data_trans = lb.transform(test_y_data)

    x = tf.placeholder(tf.float32, [None, 784])
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]))
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)
    with tf.name_scope('Wx_plus_b'):
        V = tf.matmul(x, W) + b
        tf.summary.histogram('pre_activations', V)
    with tf.name_scope('softmax'):
        y = tf.nn.softmax(V)
        tf.summary.histogram('activations', y)

    y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(loss)
    
    with tf.name_scope('evaluate'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('log/train', sess.graph)
    test_writer = tf.summary.FileWriter('log/test')
    
    if batch_gradient:
        for step in range(300):
            sess.run(train, feed_dict={x: train_x_minmax, y_: train_y_data_trans})
            if step % 10 == 0:
                print ("Batch Gradient Descent processing step %d" % step)
        print ("Finally we got the estimated results, take such a long time...")
    else:
        for step in range(1000):
            if step % 10 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={x: test_x_minmax, y_: test_y_data_trans})
                test_writer.add_summary(summary, step)
                print ("Stochastic Gradient Descent processing step %d accuracy=%.2f" % (step, acc))
            else:
                sample_index = np.random.choice(train_x_minmax.shape[0], 100)
                batch_xs = train_x_minmax[sample_index, :]
                batch_ys = train_y_data_trans[sample_index, :]
                summary, _ = sess.run([merged, train], feed_dict={x: batch_xs, y_: batch_ys})
                train_writer.add_summary(summary, step)

    np.savetxt('coef_softmax_tf.txt', np.transpose(sess.run(W)), fmt='%.6f')  # Save coefficients to a text file
    print ("Accuracy of test set: %f" % sess.run(accuracy, feed_dict={x: test_x_minmax, y_: test_y_data_trans}))
    