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
epochs = 50
steps = epochs * 550
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.InteractiveSession(config=config)
global_step = tf.Variable(0, name = 'global_step', trainable = False)
tf.set_random_seed(random_seed)

def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=trainable)

def softmax_loss(label, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

def diff(x1, x2):
    return tf.reduce_mean(tf.norm(x1 - x2, axis = 1, ord = 2))

def mean_norm(t):
    return tf.reduce_mean(tf.norm(t, axis = 1, ord = 2))

def save_images(images, size, path):
    img = images
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w] = image
    return scipy.misc.imsave(path, merge_img)

x = tf.placeholder(tf.float32, [None, 784])  # input
label = tf.placeholder(tf.float32, [None, 10])  # true label
x1 = tf.placeholder(tf.float32, [None, 784]) 

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

std = [1, 0, 0, 0]
c = [0, 0, 0, 200]
def classifier_n(x):
    n = len(layers_cls)
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = x 
            perturb = tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            #perturb_n = tf.clip_by_norm(perturb, 4, 1)
            z[i] += perturb
            z[i] = tf.clip_by_value(z[i], 0, 1)
            h[i] = z[i]
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            #z[i] = tf.clip_by_norm(z[i], c[i], 1)
            #z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = tf.nn.relu(z[i])
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w[i]) + b[i]
            #z[i] = tf.clip_by_norm(z[i], c[i], 1)
            z[i] += tf.random_normal(shape = tf.shape(z[i]), mean=0.0, stddev= std[i], dtype=tf.float32)
            h[i] = z[i]
    logits = h[n - 1]
    return logits

def accuracy(label, logits):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)),tf.float32))

def get_acc(x, label):
    y = classifier(x)
    return accuracy(label, y)

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

z1, h1 = [0 for i in range(n)], [0 for i in range(n)]
for i in range(n):
    if i == 0:
        z1[i] = x1
        h1[i] = z1[i]
    if i > 0 and i < n - 1:
        z1[i] = tf.matmul(h1[i - 1], w[i]) + b[i]
        h1[i] = tf.nn.relu(z1[i])
    if i == n - 1:
        z1[i] = tf.matmul(h1[i - 1], w[i]) + b[i]
        h1[i] = z1[i]

y = classifier(x)
yn = classifier_n(x)
loss_cls = softmax_loss(label, yn)
all_vars = tf.trainable_variables()
c_vars = [var for var in all_vars if 'classifier' in var.name]
train_op_classifier = GradientDescentOptimizer(learning_rate = learning_rate) \
        .minimize(loss_cls, var_list = c_vars, global_step = global_step)

jsma = SaliencyMapMethod(classifier, back='tf', sess=sess)
x_fgsm = attack.fgsm(x, y, eps = 0.2, clip_min=0, clip_max=1)


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        acc = {} 
        print('train classifier')
        for t in range(1, steps + 1):
            batch = mnist.train.next_batch(batch_size)
            f_dict = {x: batch[0], label: batch[1]}
            sess.run(train_op_classifier, feed_dict=f_dict)
            if t % 550 == 0:
                epoch = int(t / 550)
                acc['benign'] = sess.run(get_acc(x, label), feed_dict = {x: mnist.test.images, label: mnist.test.labels})
                print(epoch, acc)

        x_fgsm_d = sess.run(x_fgsm, feed_dict = {x: mnist.test.images, label: mnist.test.labels})
        acc['fgsm'] = sess.run(get_acc(x, label), feed_dict={x: x_fgsm_d, label: mnist.test.labels})

        y_target_batch = np.zeros((100, 10), dtype=np.float32) 
        y_target_batch[:, 0] = 1.0
        jsma_params = {'theta': 1., 'gamma': 0.1,'nb_classes': 10, 'clip_min': 0.,'clip_max': 1., 'targets': y,\
            'y_val': y_target_batch}
        x_jsma_1_d = jsma.generate_np(mnist.test.images[0:100,], **jsma_params)
        acc['jsma 1'] = sess.run(get_acc(x, label), feed_dict={x: x_jsma_1_d, label: mnist.test.labels[0:100,]})
        x_jsma_mnist_1_r = x_jsma_1_d.reshape([100, 28, 28])
        save_images(x_jsma_mnist_1_r, [10, 10], os.path.join('img', 'x_jsma_mnist_sample.png'))
        print(acc)

        print('diff fgsm')
        for i in range(4):
            print(sess.run([mean_norm(z[i]), diff(z[i], z1[i])], feed_dict = {x: mnist.test.images, x1: x_fgsm_d}))
        print('diff jsma')
        for i in range(4):
            print(sess.run([mean_norm(z[i]), diff(z[i], z1[i])], feed_dict = {x: mnist.test.images[0:100,], x1: x_jsma_1_d}))
        print('diff different image')
        for i in range(4):
            print(sess.run([mean_norm(z[i]), diff(z[i], z1[i])], feed_dict = {x: mnist.test.images[0:100,], x1: mnist.test.images[100:200,]}))

if __name__ == "__main__":
    main()
