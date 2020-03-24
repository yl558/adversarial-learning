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

# define fake image generator
noise_len = 100
layers_gan = [784 + noise_len, 1000, 784]
s = layers_gan
n = len(layers_gan)
w2, b2 = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('generator'):
    for i in range(1, n):
        w2[i] = weight_variable([s[i - 1], s[i]])
        b2[i] = bias_variable([s[i]])

def generator(x, noise, max_perturb = 2):
    n = len(layers_gan)
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
    return np.random.uniform(-1., 1., size=[m, n])

def softmax_loss(label, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

def accuracy(label, logits):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)),tf.float32))

def get_acc(x, label):
    y = classifier(x)
    return accuracy(label, y)

def diff(x1, x2):
    return tf.reduce_mean(tf.norm(x1 - x2, axis = 1))

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
    saver = tf.train.Saver()

    x = tf.placeholder(tf.float32, [None, 784])  # input
    label = tf.placeholder(tf.float32, [None, 10])  # true label
    noise = tf.placeholder(tf.float32, [None, 100]) # noise vector
    y_target = tf.placeholder(tf.float32, [None, 10]) # target label

    x1 = tf.placeholder(tf.float32, [None, 784])
    x2 = tf.placeholder(tf.float32, [None, 784])
    
    y = classifier(x)

    # gan 
    x_gan = generator(x, noise, 4)
    y_gan = classifier(x_gan)
    
    loss = softmax_loss(label, y)
    loss_gan = - softmax_loss(label, y_gan)

    all_vars = tf.trainable_variables()
    c_vars = [var for var in all_vars if 'classifier' in var.name]
    g_vars = [var for var in all_vars if 'generator' in var.name]
    train_op_classifier = GradientDescentOptimizer(learning_rate = learning_rate) \
        .minimize(loss, var_list = c_vars, global_step = global_step)
    train_op_generator = GradientDescentOptimizer(learning_rate = 0.05) \
        .minimize(loss_gan, var_list = g_vars, global_step = global_step)

    #fgsm 
    x_fgsm = attack.fgsm(x, y, eps = 0.2, clip_min=0, clip_max=1)
    y_fgsm = classifier(x_fgsm)
    # jsma
    jsma = SaliencyMapMethod(classifier, back='tf', sess=sess)
    
    # train
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    

    y_target_batch = np.zeros((100, 10), dtype=np.float32) 
    y_target_batch[:, 0] = 1.0
    y_target_test = np.zeros((10000, 10), dtype=np.float32) 
    y_target_test[:, 0] = 1.0

    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        
        print('train classifier')
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_op_classifier, feed_dict={x: batch[0], label: batch[1]})
            if t % 550 == 0:
                epoch = int(t / 550)
                acc_benign = sess.run(get_acc(x, label), feed_dict={x: mnist.test.images, label: mnist.test.labels})
                print(epoch, acc_benign)
        
        print('train gan')
        for t in range(1, 550 * 5 + 1):
            batch = mnist.train.next_batch(batch_size)
            f_dict = {x: batch[0], label: batch[1], noise: sample_Z(batch_size, 100), y_target: y_target_batch}
            sess.run(train_op_generator, feed_dict=f_dict)
            if t % 550 == 0:
                epoch = int(t / 550)
                f_dict = {x: mnist.test.images, label: mnist.test.labels, noise: sample_Z(10000, 100), y_target: y_target_batch}
                x_gan_d = sess.run(x_gan, feed_dict=f_dict)
                f_dict = {x: x_gan_d, label: mnist.test.labels}
                acc_gan = sess.run(get_acc(x, label), feed_dict=f_dict)
                print(epoch, acc_gan)

        checkpoint_path = os.path.join('model', 'basic_model.ckpt')
        #saver.save(sess, checkpoint_path, global_step = 1)


        print('generate adv samples for the first batch of the testing set')
        # real
        x_real_mnist_1 = mnist.test.images[0:100,]
        np.save(os.path.join('data','x_real_mnist_1.npy'), x_real_mnist_1)
        x_real_mnist_1_r = x_real_mnist_1.reshape([100, 28, 28])
        save_images(x_real_mnist_1_r, [10, 10], os.path.join('img', 'x_real_mnist_1.png'))
        # fgsm
        x_fgsm_mnist_1 = sess.run(x_fgsm, feed_dict = {x: mnist.test.images[0:100,], label: mnist.test.labels[0:100,]})
        np.save(os.path.join('data','x_fgsm_mnist_1.npy'), x_fgsm_mnist_1)
        x_fgsm_mnist_1_r = x_fgsm_mnist_1.reshape([100, 28, 28])
        save_images(x_fgsm_mnist_1_r, [10, 10], os.path.join('img', 'x_fgsm_mnist_1.png'))
        #jsma
        jsma_params = {'theta': 1., 'gamma': 0.1,'nb_classes': 10, 'clip_min': 0.,'clip_max': 1., 'targets': y,\
            'y_val': y_target_batch}
        x_jsma_mnist_1 = jsma.generate_np(mnist.test.images[0:100,], **jsma_params)
        np.save(os.path.join('data','x_jsma_mnist_1.npy'), x_jsma_mnist_1)
        acc_jsma_1 = sess.run(get_acc(x, label), feed_dict={x: x_jsma_mnist_1, label: mnist.test.labels[0:100,]})
        x_jsma_mnist_1_r = x_jsma_mnist_1.reshape([100, 28, 28])
        save_images(x_jsma_mnist_1_r, [10, 10], os.path.join('img', 'x_jsma_mnist_1.png'))
        
        x_gan_mnist_1 = sess.run(x_gan, feed_dict={x: mnist.test.images[0:100,] ,label: mnist.test.labels[0:100,]\
            , noise: sample_Z(batch_size, 100), y_target: y_target_batch})
        np.save(os.path.join('data','x_gan_mnist_1.npy'), x_gan_mnist_1)
        x_gan_mnist_1_r = x_gan_mnist_1.reshape([100, 28, 28])
        save_images(x_gan_mnist_1_r, [10, 10], os.path.join('img', 'x_gan_mnist_1.png'))

        diff_fgsm = sess.run(diff(x1, x2), feed_dict={x1: x_real_mnist_1, x2: x_fgsm_mnist_1})
        diff_jsma = sess.run(diff(x1, x2), feed_dict={x1: x_real_mnist_1, x2: x_jsma_mnist_1})
        diff_gan = sess.run(diff(x1, x2), feed_dict={x1: x_real_mnist_1, x2: x_gan_mnist_1})
        print('perturb: fgsm: {:.3f}, jsma: {:.3f}, gan: {:.3f}'.format(diff_fgsm, diff_jsma, diff_gan))

        acc_benign = sess.run(get_acc(x, label), feed_dict={x: mnist.test.images, label: mnist.test.labels})

        print('generate adv samples for the entire testing set')
        # fgsm
        x_fgsm_mnist = sess.run(x_fgsm, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
        np.save(os.path.join('data','x_fgsm_mnist.npy'), x_fgsm_mnist)
        acc_fgsm = sess.run(get_acc(x, label), feed_dict={x: x_fgsm_mnist, label: mnist.test.labels})
        
        # gan
        x_gan_mnist = sess.run(x_gan, feed_dict={x: mnist.test.images ,label: mnist.test.labels\
            , noise: sample_Z(10000, 100), y_target: y_target_test})
        np.save(os.path.join('data','x_gan_mnist.npy'), x_gan_mnist)
        acc_gan = sess.run(get_acc(x, label), feed_dict={x: x_gan_mnist ,label: mnist.test.labels\
            , noise: sample_Z(10000, 100), y_target: y_target_test})

        print('accuracy: benign: {:.3f}, fgsm: {:.3f}, jsma: {:.3f}, gan: {:.3f}'.format(acc_benign, acc_fgsm, acc_jsma_1, acc_gan))

        '''
        x_fgsm_mnist = np.load(os.path.join('data','x_fgsm_mnist.npy'))
        x_gan_mnist = np.load(os.path.join('data','x_gan_mnist.npy'))
        x_jsma_mnist_1 = np.load(os.path.join('data','x_jsma_mnist_1.npy'))
        sess.run(tf.global_variables_initializer())
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
        '''
        sess.close()
        return
        # jsma
        jsma_params = {'theta': 1., 'gamma': 0.1,'nb_classes': 10, 'clip_min': 0.,'clip_max': 1., 'targets': y,\
            'y_val': y_target_test}
        x_jsma_mnist = jsma.generate_np(mnist.test.images, **jsma_params)
        np.save(os.path.join('data','x_jsma_mnist.npy'), x_jsma_mnist)
        #acc_jsma = sess.run(get_acc(x, label), feed_dict={x: x_jsma_mnist, label: mnist.test.labels})
        


if __name__ == "__main__":
    main()