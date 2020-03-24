# -*- coding: utf-8 -*-
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

random_seed = 1024
train_size, test_size = 55000, 10000
batch_size = 100
learning_rate = 0.05
epochs = 1
steps = epochs * 550


def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=trainable)

# model 1
shape_1 = [784, 1000, 500, 10]
n = len(shape_1)
w_1, b_1 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('model_1'):
    for i in range(1, n):
        w_1[i] = weight_variable([shape_1[i - 1], shape_1[i]])
        b_1[i] = bias_variable([shape_1[i]])

# model 2
shape_2 = [784, 1000, 1000, 500, 500, 10]
n = len(shape_2)
w_2, b_2 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('model_2'):
    for i in range(1, n):
        w_2[i] = weight_variable([shape_2[i - 1], shape_2[i]])
        b_2[i] = bias_variable([shape_2[i]])

# model 3
shape_3 = [784, 500, 500, 500, 10]
n = len(shape_3)
w_3, b_3 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('model_3'):
    for i in range(1, n):
        w_3[i] = weight_variable([shape_3[i - 1], shape_3[i]])
        b_3[i] = bias_variable([shape_3[i]])

def model_1(x, output_bound = None):
    n = len(shape_1)
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w_1[i]) + b_1[i]
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w_1[i]) + b_1[i]
            h[i] = z[i]
    if output_bound is not None:
        h[n - 1] = tf.clip_by_norm(h[n - 1], output_bound, axes = 1)
    return h[n - 1]

def model_2(x, output_bound = None):
    n = len(shape_2)
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w_2[i]) + b_2[i]
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w_2[i]) + b_2[i]
            h[i] = z[i]
    if output_bound is not None:
        h[n - 1] = tf.clip_by_norm(h[n - 1], output_bound, axes = 1)
    return h[n - 1]

def model_3(x, output_bound = None):
    n = len(shape_3)
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w_3[i]) + b_3[i]
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w_3[i]) + b_3[i]
            h[i] = z[i]
    if output_bound is not None:
        h[n - 1] = tf.clip_by_norm(h[n - 1], output_bound, axes = 1)
    return h[n - 1]
    
def save_images(images, size, path):
    img = images
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w] = image
    return scipy.misc.imsave(path, merge_img)

def softmax_loss(label, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

def accuracy(label, logits):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)),tf.float32))

def diff(x1, x2):
    return tf.reduce_mean(tf.norm(x1 - x2, axis = 1))

def main():
    sess = tf.Session()
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    tf.set_random_seed(random_seed)
    saver = tf.train.Saver()

    x = tf.placeholder(tf.float32, [None, 784])  # input
    label = tf.placeholder(tf.float32, [None, 10])  # true label
    y_target = tf.placeholder(tf.float32, [None, 10]) # noise vector

    x1 = tf.placeholder(tf.float32, [None, 784])
    x2 = tf.placeholder(tf.float32, [None, 784])
    
    y_1 = model_1(x)
    y_2 = model_2(x)
    y_3 = model_3(x)
    
    loss_1 = softmax_loss(label, y_1)
    loss_2 = softmax_loss(label, y_2)
    loss_3 = softmax_loss(label, y_3)
    
    all_vars = tf.trainable_variables()
    model_1_vars = [var for var in all_vars if 'model_1' in var.name]
    model_2_vars = [var for var in all_vars if 'model_2' in var.name]
    model_3_vars = [var for var in all_vars if 'model_3' in var.name]
    
    train_op_1 = GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss_1, var_list = model_1_vars, global_step = global_step)
    train_op_2 = GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss_2, var_list = model_2_vars, global_step = global_step)
    train_op_3 = GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss_3, var_list = model_3_vars, global_step = global_step)
    
    x_fgsm_1 = attack.fgsm(x, y_1, eps = 0.3, clip_min=0, clip_max=1)
    x_fgsm_2 = attack.fgsm(x, y_2, eps = 0.3, clip_min=0, clip_max=1)
    x_fgsm_3 = attack.fgsm(x, y_3, eps = 0.3, clip_min=0, clip_max=1)
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    y_target = np.zeros((10000, 10), dtype=np.float32) 
    y_target[:, 0] = 1.0
    
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        print('train models')
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_op_1, feed_dict={x: batch[0], label: batch[1]})
            sess.run(train_op_2, feed_dict={x: batch[0], label: batch[1]})
            sess.run(train_op_3, feed_dict={x: batch[0], label: batch[1]})
            if t % 550 == 0:
                epoch = int(t / 550)
                acc_1 = sess.run(accuracy(label, y_1), feed_dict={x: mnist.test.images, label: mnist.test.labels})
                acc_2 = sess.run(accuracy(label, y_2), feed_dict={x: mnist.test.images, label: mnist.test.labels})
                acc_3 = sess.run(accuracy(label, y_3), feed_dict={x: mnist.test.images, label: mnist.test.labels})
                print(epoch, acc_1, acc_2, acc_3)

        print('generate adv samples')
        # fgsm
        x_fgsm_1_data = sess.run(x_fgsm_1, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
        np.save(os.path.join('data', 'mnist','x_fgsm_1_data.npy'), x_fgsm_1_data)
        acc_fgsm_1 = sess.run(accuracy(label, y_1), feed_dict={x: x_fgsm_1_data, label: mnist.test.labels})
        
        x_fgsm_2_data = sess.run(x_fgsm_2, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
        np.save(os.path.join('data', 'mnist','x_fgsm_2_data.npy'), x_fgsm_2_data)
        acc_fgsm_2 = sess.run(accuracy(label, y_2), feed_dict={x: x_fgsm_2_data, label: mnist.test.labels})
        
        x_fgsm_3_data = sess.run(x_fgsm_3, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
        np.save(os.path.join('data', 'mnist','x_fgsm_3_data.npy'), x_fgsm_3_data)
        acc_fgsm_3 = sess.run(accuracy(label, y_3), feed_dict={x: x_fgsm_3_data, label: mnist.test.labels})
                
        x_data = mnist.test.images
        
        x_perturb_data = x_data + np.random.normal(loc = 0.0, scale = 0.1, size = [10000, 784])
        x_perturb_data = np.clip(x_perturb_data, 0, 1)
        
        x_fgsm_rd_1_data = sess.run(x_fgsm_1, feed_dict = {x: x_perturb_data, label: mnist.test.labels})
        np.save(os.path.join('data', 'mnist','x_fgsm_rd_1_data.npy'), x_fgsm_rd_1_data)
        acc_fgsm_rd_1 = sess.run(accuracy(label, y_1), feed_dict={x: x_fgsm_rd_1_data, label: mnist.test.labels})

        x_fgsm_rd_2_data = sess.run(x_fgsm_2, feed_dict = {x: x_perturb_data, label: mnist.test.labels})
        np.save(os.path.join('data', 'mnist','x_fgsm_rd_2_data.npy'), x_fgsm_rd_2_data)
        acc_fgsm_rd_2 = sess.run(accuracy(label, y_2), feed_dict={x: x_fgsm_rd_2_data, label: mnist.test.labels})

        x_fgsm_rd_3_data = sess.run(x_fgsm_3, feed_dict = {x: x_perturb_data, label: mnist.test.labels})
        np.save(os.path.join('data', 'mnist','x_fgsm_rd_3_data.npy'), x_fgsm_rd_3_data)
        acc_fgsm_rd_3 = sess.run(accuracy(label, y_3), feed_dict={x: x_fgsm_rd_3_data, label: mnist.test.labels})
        
        x_fgsm_it_1_data = np.copy(mnist.test.images)
        x_fgsm_it_2_data = np.copy(mnist.test.images)
        x_fgsm_it_3_data = np.copy(mnist.test.images)
        
        for _ in range(10):
            grad_1 = sess.run(tf.gradients(loss_1, x)[0], feed_dict = {x: x_fgsm_it_1_data, label: mnist.test.labels})
            x_fgsm_it_1_data += np.sign(grad_1) * 0.01
            x_fgsm_it_1_data = np.clip(x_fgsm_it_1_data, 0, 1)
            
            grad_2 = sess.run(tf.gradients(loss_2, x)[0], feed_dict = {x: x_fgsm_it_2_data, label: mnist.test.labels})
            x_fgsm_it_2_data += np.sign(grad_2) * 0.01
            x_fgsm_it_2_data = np.clip(x_fgsm_it_2_data, 0, 1)

            grad_3 = sess.run(tf.gradients(loss_3, x)[0], feed_dict = {x: x_fgsm_it_3_data, label: mnist.test.labels})
            x_fgsm_it_3_data += np.sign(grad_3) * 0.01
            x_fgsm_it_3_data = np.clip(x_fgsm_it_3_data, 0, 1)
        
        np.save(os.path.join('data', 'mnist','x_fgsm_it_1_data.npy'), x_fgsm_it_1_data)
        acc_fgsm_it_1 = sess.run(accuracy(label, y_1), feed_dict={x: x_fgsm_it_1_data, label: mnist.test.labels})
        
        np.save(os.path.join('data', 'mnist','x_fgsm_it_2_data.npy'), x_fgsm_it_2_data)
        acc_fgsm_it_2 = sess.run(accuracy(label, y_2), feed_dict={x: x_fgsm_it_2_data, label: mnist.test.labels})
        
        np.save(os.path.join('data', 'mnist','x_fgsm_it_3_data.npy'), x_fgsm_it_3_data)
        acc_fgsm_it_3 = sess.run(accuracy(label, y_3), feed_dict={x: x_fgsm_it_3_data, label: mnist.test.labels})
        
        
        grad_tg_1 = sess.run(tf.gradients(loss_1, x)[0], feed_dict = {x: mnist.test.images, label: y_target})
        x_fgsm_tg_1_data = mnist.test.images - np.sign(grad_tg_1) * 0.3
        x_fgsm_tg_1_data = np.clip(x_fgsm_tg_1_data, 0, 1)
        
        grad_tg_2 = sess.run(tf.gradients(loss_2, x)[0], feed_dict = {x: mnist.test.images, label: y_target})
        x_fgsm_tg_2_data = mnist.test.images - np.sign(grad_tg_2) * 0.3
        x_fgsm_tg_2_data = np.clip(x_fgsm_tg_2_data, 0, 1)
        
        grad_tg_3 = sess.run(tf.gradients(loss_3, x)[0], feed_dict = {x: mnist.test.images, label: y_target})
        x_fgsm_tg_3_data = mnist.test.images - np.sign(grad_tg_3) * 0.3
        x_fgsm_tg_3_data = np.clip(x_fgsm_tg_3_data, 0, 1)
        
        np.save(os.path.join('data', 'mnist','x_fgsm_tg_1_data.npy'), x_fgsm_tg_1_data)
        acc_fgsm_tg_1 = sess.run(accuracy(label, y_1), feed_dict={x: x_fgsm_tg_1_data, label: mnist.test.labels})
        
        np.save(os.path.join('data', 'mnist','x_fgsm_tg_2_data.npy'), x_fgsm_tg_2_data)
        acc_fgsm_tg_2 = sess.run(accuracy(label, y_2), feed_dict={x: x_fgsm_tg_2_data, label: mnist.test.labels})

        np.save(os.path.join('data', 'mnist','x_fgsm_tg_3_data.npy'), x_fgsm_tg_3_data)
        acc_fgsm_tg_3 = sess.run(accuracy(label, y_3), feed_dict={x: x_fgsm_tg_3_data, label: mnist.test.labels})


        print('Accuracy fgsm bl: {:.4f}, {:.4f}, {:.4f}'.format(acc_fgsm_1, acc_fgsm_2, acc_fgsm_3))
        print('Accuracy fgsm rd: {:.4f}, {:.4f}, {:.4f}'.format(acc_fgsm_rd_1, acc_fgsm_rd_2, acc_fgsm_rd_3))
        print('Accuracy fgsm it: {:.4f}, {:.4f}, {:.4f}'.format(acc_fgsm_it_1, acc_fgsm_it_2, acc_fgsm_it_3))
        print('Accuracy fgsm tg: {:.4f}, {:.4f}, {:.4f}'.format(acc_fgsm_tg_1, acc_fgsm_tg_2, acc_fgsm_tg_3))
        
    return



if __name__ == "__main__":
    main()