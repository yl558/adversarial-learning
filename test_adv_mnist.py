import os, sys, random, time, math
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import keras
from cleverhans.utils_tf import model_train, model_eval
import cleverhans.attacks_tf as attack
from cleverhans.attacks import SaliencyMapMethod
import scipy.misc


def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=trainable)

# define classifier
layers_cls = [784, 1000, 500, 10]
n = len(layers_cls)
w, b = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('classifier'):
    for i in range(1, n):
        w[i] = weight_variable([layers_cls[i - 1], layers_cls[i]])
        b[i] = bias_variable([layers_cls[i]])

def classifier(x):
    n = len(layers_cls)
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            h[i] = z[i]
    logits = h[n - 1]
    return logits


def data_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, Y_train, X_test, Y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    return X_train, Y_train, X_test, Y_test

def softmax_loss(label, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

def accuracy(label, logits):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)),tf.float32))

def get_acc(x, label):
    y = classifier(x)
    return accuracy(label, y)


def main():
    random_seed = 1024
    train_size, test_size = 55000, 10000
    batch_size = 100 
    learning_rate = 0.05
    epochs = 20
    steps = epochs * 550
    sess = tf.Session()
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    tf.set_random_seed(random_seed)
    

    x = tf.placeholder(tf.float32, [None, 784])  # input
    label = tf.placeholder(tf.float32, [None, 10])  # true label
    
    y = classifier(x)
    loss_cls = softmax_loss(label, y)

    all_vars = tf.trainable_variables()
    print(all_vars)
    c_vars = [var for var in all_vars if 'classifier' in var.name]
    train_op_classifier = GradientDescentOptimizer(learning_rate = learning_rate) \
        .minimize(loss_cls, var_list = c_vars, global_step = global_step)
    
    
    # train
    saver = tf.train.Saver()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_fgsm_mnist = np.load(os.path.join('data','x_fgsm_mnist.npy'))
    x_gan_mnist = np.load(os.path.join('data','x_gan_mnist.npy'))
    x_jsma_mnist_1 = np.load(os.path.join('data','x_jsma_mnist_1.npy'))
    
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        
        print('train classifier')
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_op_classifier, feed_dict={x: batch[0], label: batch[1]})
            if t % 550 == 0:
                epoch = int(t / 550)
                acc = {}
                acc['benign'] = sess.run(get_acc(x, label), feed_dict = {x: mnist.test.images, label: mnist.test.labels})
                acc['pre fgsm'] = sess.run(get_acc(x, label), feed_dict={x: x_fgsm_mnist, label: mnist.test.labels})
                acc['pre gan'] = sess.run(get_acc(x, label), feed_dict={x: x_gan_mnist, label: mnist.test.labels})
                acc['pre jsma 1'] = sess.run(get_acc(x, label), feed_dict={x: x_jsma_mnist_1, label: mnist.test.labels[0:100,]})
                print(epoch, acc)
    
    sess.close()

if __name__ == "__main__":
    main()