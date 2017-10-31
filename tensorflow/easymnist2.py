# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:20:03 2017

@author: xing
"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#初始化权值W，b的各个参数为0
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

#预测的概率分布
y = tf.nn.softmax(tf.matmul(x,W) + b)

#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
for i in range(1000):
    #让模型循环训练1000次
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    
#得到一组布尔值,为了确定正确预测项的比例
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#把布尔值转换成浮点数，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#计算所学习到的模型在测试数据集上面的正确率
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))