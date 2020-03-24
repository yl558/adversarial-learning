import os, sys, random, time, math
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import cleverhans.attacks_tf as attack
import scipy.misc

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=trainable)

shape = {}
shape['generator'] = [100, 1000, 784]
s = shape['generator']
n = len(s)
w2, b2 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('generator'):
    for i in range(1, n):
        w2[i] = weight_variable([s[i - 1] + 10, s[i]])
        b2[i] = bias_variable([s[i]])

shape['discriminator'] = [784, 500, 1]
s = shape['discriminator']
n = len(s)
w3, b3 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('discriminator'):
    for i in range(1, n):
        w3[i] = weight_variable([s[i - 1] + 10, s[i]])
        b3[i] = bias_variable([s[i]])

def generator(noise, y):
    n = len(shape['generator'])
    s = shape['generator']
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = noise 
            z[i] = tf.concat([z[i], y], 1)
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w2[i]) + b2[i]
            #h[i] = tf.nn.relu(z[i])
            h[i] = tf.concat([tf.nn.relu(z[i]), y], 1)
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w2[i]) + b2[i]
            h[i] = tf.nn.sigmoid(z[i])
    out = h[n - 1]
    return out

def discriminator(x, y):
    n = len(shape['discriminator'])
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            h[i] = tf.concat([z[i], y], 1)
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w3[i]) + b3[i]
            #h[i] = tf.nn.relu(z[i])
            h[i] = tf.concat([tf.nn.relu(z[i]), y], 1)
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w3[i]) + b3[i]
            h[i] = tf.nn.sigmoid(z[i])
    out = h[n - 1]
    return out

def data_mnist():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train, Y_train, X_test, Y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    return X_train, Y_train, X_test, Y_test

def save_images(images, size, path):
    img = images
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w] = image
    return scipy.misc.imsave(path, merge_img)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

def main():
    train_size, test_size = 55000, 10000
    batch_size = 100
    learning_rate = 0.05
    epochs = 40
    steps = epochs * int(train_size / batch_size)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    x = tf.placeholder(tf.float32, [None, 784]) # input for real images
    noise = tf.placeholder(tf.float32, [None, 100])
    y = tf.placeholder(tf.float32, [None, 10])

    d_label_real = tf.ones(shape = [tf.shape(x)[0], 1]) # groundtruth discriminator label for real images
    d_label_fake = tf.zeros(shape = [tf.shape(x)[0], 1]) # groundtruth discriminator label for fake images

    x_fake = generator(noise, y)
    d_real = discriminator(x, y)
    d_fake = discriminator(x_fake, y)
    d_real_mean = tf.reduce_mean(d_real)
    d_fake_mean = tf.reduce_mean(d_fake)

    loss = {}
    loss['discriminator'] = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
    loss['generator'] = -tf.reduce_mean(tf.log(d_fake))

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'generator' in var.name]
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    
    train_op = {}
    train_op['generator'] = GradientDescentOptimizer(learning_rate = learning_rate) \
        .minimize(loss['generator'], var_list = g_vars, global_step = global_step)
    train_op['discriminator'] = GradientDescentOptimizer(learning_rate = learning_rate) \
        .minimize(loss['discriminator'], var_list = d_vars, global_step = global_step)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    tf.set_random_seed(1024)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_op['discriminator'], feed_dict={x: batch[0], y: batch[1], noise: sample_Z(batch_size, 100)})
            for j in range(5):
                sess.run(train_op['generator'], feed_dict={x: batch[0], y: batch[1], noise: sample_Z(batch_size, 100)})
    
            if t % int(train_size / batch_size) == 0:
                epoch = int(t / int(train_size / batch_size))
                var_list = [d_real_mean, d_fake_mean]
                res = sess.run(var_list, feed_dict = {x: batch[0], y: batch[1], noise: sample_Z(batch_size, 100)})
                print(epoch)
                for r in res:
                    print(r)
                
                if epoch == epochs:
                    img_real = batch[0].reshape([batch_size, 28, 28])
                    save_images(img_real, [10, 10], os.path.join('img', 'real.png'))
                    img_fake = sess.run(x_fake, feed_dict = {x: batch[0], y: batch[1], noise: sample_Z(batch_size, 100)})
                    img_fake = img_fake.reshape([batch_size, 28, 28])
                    save_images(img_fake, [10, 10], os.path.join('img', 'fake.png'))
                
if __name__ == "__main__":
    main()