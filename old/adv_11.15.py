import os, sys, random, time, math
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import keras
from gaussian_moments import *
from cleverhans.utils_tf import model_train, model_eval
import cleverhans.attacks_tf as attack

layers = [784, 1000, 1000, 500, 500, 10]
random_seed = 1024
sample_rate = 0.002
train_size, test_size = 55000, 10000
batch_size = int(train_size * 0.002) # 110
learning_rate = 0.005
epochs = 300
steps = epochs * int(1 / sample_rate) 

# compute sigma using strong composition theory given epsilon
def compute_sigma(epsilon, delta):
    s = np.log(np.sqrt(2/np.pi)/delta)
    sigma = np.sqrt(2) / (2 * epsilon) * (np.sqrt(s) + np.sqrt(s + delta))
    return sigma

def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=trainable)

def data_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, Y_train, X_test, Y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    return X_train, Y_train, X_test, Y_test

def model_train(para):
    sess = tf.Session()
    tf.set_random_seed(random_seed)
    n = len(layers)
    x = tf.placeholder(tf.float32, [None, 784])  # input
    label = tf.placeholder(tf.float32, [None, 10])  # true label
    std = para['std']

    w, b = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(1, n):
        w[i] = weight_variable([layers[i - 1], layers[i]])
        b[i] = bias_variable([layers[i]])
    
    # model with noise
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            z[i] = tf.clip_by_value(z[i], 0, 1)
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            #z[i] = tf.clip_by_norm(z[i], 1, axes = 1)
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            #z[i] = tf.clip_by_norm(z[i], 1000, axes = 1)
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = z[i]
    y = h[n - 1]

    w_sum = tf.constant(0, dtype = 'float32')
    for i in range(1, n):
        w_sum += tf.reduce_sum(tf.square(w[i]))

    # gradient descent
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y)) 
    gw, gb = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(1, n):
        gw[i] = tf.gradients(loss, w[i])[0]
        gb[i] = tf.gradients(loss, b[i])[0]
    opt = GradientDescentOptimizer(learning_rate = learning_rate)
    gradients = []
    for i in range(1, n):
        gradients.append((gw[i], w[i]))
        gradients.append((gb[i], b[i]))
    train_step = opt.apply_gradients(gradients)

    # model without noise
    z2, h2 = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z2[i] = x 
            h2[i] = z2[i]
        if i > 0 and i < n - 1:
            z2[i] = tf.matmul(h2[i - 1], w[i]) + b[i]
            h2[i] = tf.nn.relu(z2[i])
        if i == n - 1:
            z2[i] = tf.matmul(h2[i - 1], w[i]) + b[i]
            h2[i] = z2[i]
    y2 = h2[n - 1]
    
    # attack
    x_adv = attack.fgsm(x, y2, eps = 0.3, clip_min = 0, clip_max = 1)
    
    #evaluation
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y2, 1), tf.argmax(label, 1)),tf.float32))
    
    # data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_adv_mnist_fsgm = np.load(os.path.join('data','x_adv_mnist_fsgm.npy'))

    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch[0], label: batch[1]})
            if t % int(1 / sample_rate) == 0:
                epoch = int(t / int(1 / sample_rate))

                x_adv_sample = sess.run(x_adv, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
                acc_benign = sess.run(acc, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
                acc_adv = sess.run(acc, feed_dict = {x: x_adv_sample, label: mnist.test.labels})
                acc_pre_adv = sess.run(acc, feed_dict = {x: x_adv_mnist_fsgm, label: mnist.test.labels})
                print(epoch, acc_benign, acc_adv, acc_pre_adv)
                check = tf.reduce_mean(tf.norm(y2, axis = 1))
                print(sess.run([check], feed_dict = {x: mnist.test.images, label: mnist.test.labels}))
def main():
    para = {}
    para['std'] = [1, 2, 0, 0, 0, 100]
    model_train(para)
if __name__ == "__main__":
    main()