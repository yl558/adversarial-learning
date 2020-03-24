import os, sys, random, time, math
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import cleverhans.attacks_tf as attack
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

shape['noise'] = 100
shape['transformer'] = [784 + shape['noise'], 1000, 784]
s = shape['transformer']
n = len(s)
w2, b2 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('transformer'):
    for i in range(1, n):
        w2[i] = weight_variable([s[i - 1], s[i]])
        b2[i] = bias_variable([s[i]])

def transformer(x, noise, max_perturb = 0):
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
            h[i] = tf.nn.sigmoid(z[i])
            #h[i] = tf.nn.tanh(z[i])
    #h[n - 1] = tf.clip_by_norm(h[n - 1], thr, axes = 1)
    #x1 = x + h[n - 1]
    #x1 = tf.clip_by_value(x1, 0, 1)
    #out = x1
    out = h[n - 1]
    return out


shape['discriminator'] = [784, 1000, 1]
s = shape['discriminator']
n = len(s)
w3, b3 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('discriminator'):
    for i in range(1, n):
        w3[i] = weight_variable([s[i - 1] + 10, s[i]])
        b3[i] = bias_variable([s[i]])

def discriminator(x, y):
    n = len(shape['discriminator'])
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            h[i] = tf.concat([z[i], y], 1)
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w3[i]) + b3[i]
            h[i] = tf.concat([tf.nn.relu(z[i]), y], 1)
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w3[i]) + b3[i]
            h[i] = tf.nn.sigmoid(z[i])
    out = h[n - 1]
    return out

shape['generator'] = [100, 1000, 784]
s = shape['generator']
n = len(s)
w4, b4 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('generator'):
    for i in range(1, n):
        w4[i] = weight_variable([s[i - 1] + 10, s[i]])
        b4[i] = bias_variable([s[i]])

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
            z[i] = tf.matmul(h[i - 1], w4[i]) + b4[i]
            h[i] = tf.concat([tf.nn.relu(z[i]), y], 1)
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w4[i]) + b4[i]
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
    lr = {}
    lr['c'] = 0.05
    lr['d'] = 0.05
    lr['g'] = 0.05
    lr['t'] = 0.05
    epochs = 30
    steps = epochs * int(train_size / batch_size)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    x = tf.placeholder(tf.float32, [None, 784]) # input for real images
    y_ = tf.placeholder(tf.float32, [None, 10]) # groundtruth class label 
    y_target = tf.placeholder(tf.float32, [None, 10])
    noise = tf.placeholder(tf.float32, [None, 100])

    tx = transformer(x, noise)
    gx = generator(noise, y_)
    yx = classifier(x)
    ytx = classifier(tx)
    ygx = classifier(gx)
    dx = discriminator(x, y_)
    dtx = discriminator(tx, y_)
    dgx = discriminator(gx, y_)

    d_mean = {}
    d_mean['x'] = tf.reduce_mean(dx)
    d_mean['tx'] = tf.reduce_mean(dtx)
    d_mean['gx'] = tf.reduce_mean(dgx)

    #x_fgsm = attack.fgsm(x, y_real, eps = 0.1, clip_min = 0, clip_max = 1)
    #c_x_fgsm = classifier(x_fgsm)
    #d_fg = discriminator(x_fg)

    perturb = {}
    perturb['tx'] = tf.reduce_mean(tf.norm(tx - x, axis = 1))
    perturb['gx'] = tf.reduce_mean(tf.norm(gx - x, axis = 1))
    #perturb['fg'] = tf.reduce_mean(tf.norm(x_fg - x, axis = 1))

    loss = {}
    loss['cx'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=yx))
    loss['ctx'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=ytx))
    loss['cgx'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=ygx))

    loss['ctgx'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=ygx))
    loss['cttx'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=ytx))

    #loss['classifier_fgsm'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y_fg))

    loss['dx'] = tf.reduce_mean( - tf.log(dx))
    loss['dtx'] = tf.reduce_mean( - tf.log(1. - dtx) )
    loss['dgx'] = tf.reduce_mean( - tf.log(1. - dgx) )

    loss['t'] = tf.reduce_mean( - tf.log(dtx))
    loss['g'] = tf.reduce_mean( - tf.log(dgx))

    loss['classifier'] = loss['cx'] 
    loss['transformer'] = loss['t'] + loss['cttx'] 
    loss['generator'] = loss['g'] 
    loss['discriminator'] = loss['dx'] + loss['dtx']


    all_vars = tf.trainable_variables()
    c_vars = [var for var in all_vars if 'classifier' in var.name]
    g_vars = [var for var in all_vars if 'generator' in var.name]
    d_vars = [var for var in all_vars if 'discriminator' in var.name]
    t_vars = [var for var in all_vars if 'transformer' in var.name]
    
    train_op = {}
    train_op['classifier'] = GradientDescentOptimizer(learning_rate = lr['c']) \
        .minimize(loss['classifier'], var_list = c_vars, global_step = global_step)
    train_op['generator'] = GradientDescentOptimizer(learning_rate = lr['g']) \
        .minimize(loss['generator'], var_list = g_vars, global_step = global_step)
    train_op['discriminator'] = GradientDescentOptimizer(learning_rate = lr['d']) \
        .minimize(loss['discriminator'], var_list = d_vars, global_step = global_step)
    train_op['transformer'] = GradientDescentOptimizer(learning_rate = lr['t']) \
        .minimize(loss['transformer'], var_list = t_vars, global_step = global_step)

    acc = {}
    acc['x'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yx, 1), tf.argmax(y_, 1)),tf.float32))
    acc['tx'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ytx, 1), tf.argmax(y_, 1)),tf.float32))
    acc['gx'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ygx, 1), tf.argmax(y_, 1)),tf.float32))
    #acc['classifier_fgsm'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_fg, 1), tf.argmax(label, 1)),tf.float32))


    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    tf.set_random_seed(1024)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            y_tar = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(batch_size)]
            y_tar = np.array(y_tar, dtype = 'float32')

            noise_sample = sample_Z(batch_size, 100)

            sess.run(train_op['classifier'], feed_dict={x: batch[0], y_: batch[1], noise : noise_sample, y_target: y_tar})
            sess.run(train_op['discriminator'], feed_dict={x: batch[0], y_: batch[1], noise : noise_sample, y_target: y_tar})
            for j in range(5):
                sess.run(train_op['transformer'], feed_dict={x: batch[0], y_: batch[1], noise : noise_sample, y_target: y_tar})
            #for j in range(5):
                #sess.run(train_op['generator'], feed_dict={x: batch[0], y_: batch[1], noise : sample_Z(batch_size, 100), y_target: y_tar})
            
            if t % int(train_size / batch_size) == 0:
                epoch = int(t / int(train_size / batch_size))

                noise_sample3 = sample_Z(batch_size, 100)
                batch = mnist.test.next_batch(100)
                img_real = batch[0].reshape([100, 28, 28])
                save_images(img_real, [10, 10], os.path.join('img', 'real' + str(epoch) + '.png'))

                img_trans = sess.run(tx, feed_dict = {x: batch[0], y_: batch[1], noise : noise_sample3, y_target: y_tar})
                img_trans = img_trans.reshape([100, 28, 28])
                save_images(img_trans, [10, 10], os.path.join('img', 'trans' + str(epoch) + '.png'))
                
                img_fake = sess.run(gx, feed_dict = {x: batch[0], y_: batch[1], noise : noise_sample3, y_target: y_tar})
                img_fake = img_fake.reshape([100, 28, 28])
                save_images(img_fake, [10, 10], os.path.join('img', 'fake' + str(epoch) + '.png'))


                noise_sample2 = sample_Z(10000, 100)
                var_list = [acc, perturb, d_mean]
                res = sess.run(var_list, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, noise : noise_sample2, y_target: y_tar})
                print(epoch)
                for r in res:
                    print(r)
                

        

                
if __name__ == "__main__":
    main()