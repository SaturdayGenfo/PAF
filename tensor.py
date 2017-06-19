# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:11:19 2017


"""

import tensorflow as tf

from dataformat import mnist, get_data, get_down_sampled_batch

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#model
y = tf.nn.softmax(tf.matmul(x, W) + b)

ycorr = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ycorr, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.95).minimize(cross_entropy)


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


#batch_xs, batch_ys = get_data(1, mnist.train)

for i in range(1000):
    bxs, bys = get_down_sampled_batch(1, 100)
    sess.run(train_step, feed_dict={x : bxs, ycorr : bys})
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(ycorr, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

a, b = get_data(1, mnist.test)
print(sess.run(accuracy, feed_dict={x:a, ycorr: b}))