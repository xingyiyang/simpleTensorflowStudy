# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:20:03 2017

@author: xing
"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

#通过操作符号变量来描述这些可交互的操作单元
x = tf.placeholder(tf.float32, [None, 784])

#初始化权值W，b的各个参数为0
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#预测的概率分布
y = tf.nn.softmax(tf.matmul(x,W) + b)

#实际分布
y_ = tf.placeholder("float", [None,10])

#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#添加一个操作来初始化我们创建的变量
init = tf.global_variables_initializer()

#在一个Session里面启动我们的模型
sess=tf.Session()
sess.run(init)
    
for i in range(1000):
    #让模型循环训练1000次
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
#得到一组布尔值,为了确定正确预测项的比例
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#把布尔值转换成浮点数，然后取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))