import os, sys, random, time, math
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import cleverhans.attacks_tf as attack
from cleverhans.attacks import SaliencyMapMethod
import scipy.misc

def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=trainable)

shape = {}
shape['classifier'] = [784, 1000, 500, 10]
s = shape['classifier']
n = len(s)
w, b = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('classifier'):
    for i in range(1, n):
        w[i] = weight_variable([s[i - 1], s[i]])
        b[i] = bias_variable([s[i]])

def classifier(x):
    n = len(shape['classifier'])
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

std = [0, 0, 0, 0]
bound = [10*10, 25*10, 40*10, 15*10]

def classifier_x(x):
    n = len(shape['classifier'])
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            z[i] = tf.clip_by_norm(z[i], bound[i], 1)
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            #z[i] = tf.clip_by_norm(z[i], bound[i], 1)
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            z[i] = tf.clip_by_norm(z[i], bound[i], 1)
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = z[i]
    logits = h[n - 1]
    z_norm = [0 for i in range(n)]
    for i in range(n):
        z_norm[i] = tf.reduce_mean(tf.norm(z[i], axis = 1))
    return logits, z_norm


shape['noise'] = 100
shape['transformer'] = [784 + shape['noise'], 1000, 784]
s = shape['transformer']
n = len(s)
w2, b2 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('transformer'):
    for i in range(1, n):
        w2[i] = weight_variable([s[i - 1], s[i]])
        b2[i] = bias_variable([s[i]])

def transformer(x, noise, max_perturb = 2):
    n = len(shape['transformer'])
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            z[i] = tf.concat([z[i], noise], 1)
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w2[i]) + b2[i]
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w2[i]) + b2[i]
            h[i] = tf.nn.tanh(z[i])
    h[n - 1] = tf.clip_by_norm(h[n - 1], max_perturb, axes = 1)
    x1 = x + h[n - 1]
    x1 = tf.clip_by_value(x1, 0, 1)
    out = x1
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
    sess = tf.Session()
    train_size, test_size = 55000, 10000
    batch_size = 100 
    lr = 0.05
    epochs = 100
    steps = epochs * int(train_size / batch_size)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    x = tf.placeholder(tf.float32, [None, 784]) # input for real images
    x_adv = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10]) # groundtruth class label 
    y_target = tf.placeholder(tf.float32, [None, 10])
    noise = tf.placeholder(tf.float32, [None, 100])

    tx = transformer(x, noise, 3)
    yx_2, z_norm = classifier_x(x)
    yx = classifier(x)
    ytx = classifier(tx)
    ytx_2, z_norm2 = classifier_x(tx)
    y_x_adv = classifier(x_adv)

    x_fgsm = attack.fgsm(x, yx_2, eps = 0.1, clip_min = 0, clip_max = 1)
    y_x_fgsm = classifier(x_fgsm)

    jsma = SaliencyMapMethod(classifier, back='tf', sess=sess)
    one_hot_target = np.zeros((100, 10), dtype=np.float32) 
    one_hot_target[:, 1] = 1
    jsma_params = {'theta': 1., 'gamma': 0.1,
                           'nb_classes': 10, 'clip_min': 0.,
                           'clip_max': 1., 'targets': yx,
                           'y_val': one_hot_target}

    perturb = {}
    perturb['tx'] = tf.reduce_mean(tf.norm(tx - x, axis = 1))
    perturb['fgsm'] = tf.reduce_mean(tf.norm(x_fgsm - x, axis = 1))

    loss = {}
    loss['cx'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=yx_2))
    loss['ctx'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=ytx_2))

    loss['cttx'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=ytx))

    loss['classifier'] = loss['cx'] #+ loss['ctx']
    #loss['transformer'] = loss['cttx'] 
    loss['transformer'] = - loss['ctx']

    all_vars = tf.trainable_variables()
    c_vars = [var for var in all_vars if 'classifier' in var.name]
    t_vars = [var for var in all_vars if 'transformer' in var.name]

    train_op = {}
    train_op['classifier'] = GradientDescentOptimizer(learning_rate = lr) \
        .minimize(loss['classifier'], var_list = c_vars, global_step = global_step)
    train_op['transformer'] = GradientDescentOptimizer(learning_rate = lr) \
        .minimize(loss['transformer'], var_list = t_vars, global_step = global_step)

    acc = {}
    acc['x'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yx, 1), tf.argmax(y_, 1)),tf.float32))
    acc['tx'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ytx, 1), tf.argmax(y_, 1)),tf.float32))
    acc['x_fgsm'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_x_fgsm, 1), tf.argmax(y_, 1)),tf.float32))
    acc['x_adv'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_x_adv, 1), tf.argmax(y_, 1)),tf.float32))

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x_adv_mnist_fsgm = np.load(os.path.join('data','x_fgsm_mnist.npy'))
    tf.set_random_seed(1024)
    
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            y_tar = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(batch_size)]
            y_tar = np.array(y_tar, dtype = 'float32')

            noise_sample = sample_Z(batch_size, 100)

            sess.run(train_op['classifier'], feed_dict={x: batch[0], y_: batch[1], noise : noise_sample, y_target: y_tar})
            sess.run(train_op['transformer'], feed_dict={x: batch[0], y_: batch[1], noise : noise_sample, y_target: y_tar})
        
            
            if t % int(train_size / batch_size) == 0:
                epoch = int(t / int(train_size / batch_size))

                noise_sample2 = sample_Z(10000, 100)
                test_batch = mnist.test.next_batch(10000)
                print(test_batch[0].shape)
                var_list = [acc, z_norm]
                res = sess.run(var_list, feed_dict = {x: test_batch[0], y_: test_batch[1], noise : noise_sample2, \
                    x_adv: x_adv_mnist_fsgm, y_target: y_tar})
                print(epoch)
                for r in res:
                    print(r)

                #x_jsma = jsma.generate_np(test_batch[0], **jsma_params)
                #print(x_jsma.shape)
                
if __name__ == "__main__":
    main()