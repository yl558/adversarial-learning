import os
import numpy as np
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import scipy

random_seed = 1024
tf.set_random_seed(random_seed)
sess = tf.Session()
lr = 0.05
epochs = 5
epochs2 = 20
batch_size = 100
steps = 10000
global_step = tf.Variable(0, name = 'global_step', trainable = False)


def weight_variable(shape, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)

def bias_variable(shape, trainable=True):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=trainable)

shape = {}
shape['generator'] = [100, 1000, 1000, 784]
s = shape['generator']
n = len(s)
w_g, b_g = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('generator'):
    for i in range(1, n):
        w_g[i] = weight_variable([s[i - 1] + 10, s[i]])
        b_g[i] = bias_variable([s[i]])

def generator(noise, y):
    n = len(shape['generator'])
    s = shape['generator']
    z, h = [0 for i in range(n)], [0 for i in range(n)]
    for i in range(n):
        if i == 0:
            z[i] = noise 
            h[i] = tf.concat([z[i], y], 1)
        if i > 0 and i < n - 1:
            z[i] = tf.matmul(h[i - 1], w_g[i]) + b_g[i]
            h[i] = tf.nn.relu(z[i])
            h[i] = tf.concat([tf.nn.relu(h[i]), y], 1)
        if i == n - 1:
            z[i] = tf.matmul(h[i - 1], w_g[i]) + b_g[i]
            h[i] = tf.nn.sigmoid(z[i])
    return h[n - 1]

# define classifier
shape_cls = [784, 1000, 500, 10]
n = len(shape_cls)
w, b = [0 for i in range(n)], [0 for i in range(n)]
with tf.variable_scope('classifier'):
    for i in range(1, n):
        w[i] = weight_variable([shape_cls[i - 1], shape_cls[i]])
        b[i] = bias_variable([shape_cls[i]])

def classifier(x):
    n = len(shape_cls)
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
    return  h[n - 1]

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

def softmax_cross_entropy(label, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

def accuracy(label, logits):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)),tf.float32))

def get_acc(x, label):
    y = classifier(x)
    return accuracy(label, y)

x = tf.placeholder(tf.float32, [None, 784])
y = classifier(x)
label = tf.placeholder(tf.float32, [None, 10])  # true label
noise = tf.placeholder(tf.float32, [None, 100]) # noise vector

x_fake = generator(noise, label)
y_fake = classifier(x_fake)

loss_cls = softmax_cross_entropy(label, y)
loss_gan = softmax_cross_entropy(label, y_fake)

all_vars = tf.trainable_variables()
c_vars = [var for var in all_vars if 'classifier' in var.name]
g_vars = [var for var in all_vars if 'generator' in var.name]
train_op_cls = GradientDescentOptimizer(learning_rate = lr) \
    .minimize(loss_cls, var_list = c_vars, global_step = global_step)
train_op_gan = GradientDescentOptimizer(learning_rate = lr) \
    .minimize(loss_gan, var_list = g_vars, global_step = global_step)

def main():
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        print('train classifier')
        for t in range(1, epochs * 550 + 1):
            batch = mnist.train.next_batch(batch_size)
            f_dict = {x: batch[0], label: batch[1]}
            sess.run(train_op_cls, feed_dict=f_dict)
            if t % 550 == 0:
                epoch = int(t / 550)
                acc = sess.run(get_acc(x, label), feed_dict = {x: mnist.test.images, label: mnist.test.labels})
                print(acc)

        print('train gan')
        for t in range(1, epochs2 * 550 + 1):
            batch_noise = sample_Z(batch_size, 100)
            batch = mnist.train.next_batch(batch_size)
            f_dict = {noise: batch_noise, label: batch[1]}
            sess.run(train_op_gan, feed_dict=f_dict)
            if t % 550 == 0:
                epoch = int(t / 550)
                batch_noise = sample_Z(100, 100)
                batch = mnist.test.next_batch(batch_size)
                x_fake_d = sess.run(x_fake, feed_dict = {noise: batch_noise, label: batch[1]})
                acc = sess.run(get_acc(x, label), feed_dict = {x: x_fake_d, label: batch[1]})
                print(acc)
                #if t == epochs2 * 550 + 1:
                x_fake_d = x_fake_d.reshape([100, 28, 28])
                save_images(x_fake_d, [10, 10], os.path.join('img', 'fake_img_1.png'))

if __name__ == "__main__":
    main()