# Author : hellcat
# Time   : 18-4-19

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""

import tensorflow as tf


def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        x_padded = tf.pad(x, [[0, 0],
                              [int(kernel / 2), int(kernel / 2)],
                              [int(kernel / 2), int(kernel / 2)],
                              [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def conv2d_transpose(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv_transpose'):
        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides

        output_shape = tf.stack([batch_size, height, width, output_filters])
        return tf.nn.conv2d_transpose(x, weight, output_shape,
                                      strides=[1, strides, strides, 1], name='conv_transpose')


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    """
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/
    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    原先我们需要使用网络生成图像的时候，一般都是采用转置卷积直接对图像进行上采样。
    一篇文章指出了转置卷积的一些问题，认为转置卷积由于不合理的重合，使得生成的图片总是有“棋盘状的噪声点”，
    它提出使用先将图片放大，再做卷积的方式来代替转置卷积做上采样，可以提高生成图片的质量
    """
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        # 乘stride为了抵消卷积层的下采样，乘2为了实现上采样
        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x,
                                           [new_height, new_width],
                                           tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)


def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def batch_norm(x, size, training, decay=0.999):
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    pop_mean = tf.Variable(tf.zeros([size]))
    pop_var = tf.Variable(tf.ones([size]))
    epsilon = 1e-3

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch_norm')

    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon, name='batch_norm')

    return tf.cond(training, batch_statistics, population_statistics)


def relu(input_):
    re = tf.nn.relu(input_)
    # convert nan to zero (nan != nan)
    nan_to_zero = tf.where(tf.equal(re, re), re, tf.zeros_like(re))
    return nan_to_zero


def residual(x, filters, kernel, strides):
    """
    保证输入步长strides为1，防止图像降采样,也正因如此，此处没有设置shortcut（对x使用等步长卷积降采样）
    """
    with tf.variable_scope('residual'):
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(relu(conv1), filters, filters, kernel, strides)
        res = x + conv2
        return res


def net(image, training):
    """
    填充——>卷积——>下采样卷积×2——>参差网络×5——>上采样卷积×2——>卷积——>剔除填充
    :param image: 输入图片tensor
    :param training: True or False
    :return: 重建图片
    """
    # Less border effects when padding a little before passing through ..
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    with tf.variable_scope('conv1'):
        conv1 = relu(instance_norm(conv2d(image, 3, 32, 9, 1)))
    with tf.variable_scope('conv2'):
        conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))
    with tf.variable_scope('conv3'):
        conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    # print(res5.get_shape())
    with tf.variable_scope('deconv1'):
        # deconv1 = relu(instance_norm(conv2d_transpose(res5, 128, 64, 3, 2)))
        deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        # deconv2 = relu(instance_norm(conv2d_transpose(deconv1, 64, 32, 3, 2)))
        deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        # deconv3 = conv2d(deconv2, 32, 3, 9, 1)
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    y = (deconv3 + 1) * 127.5  # Tensor("mul:0", shape=(4, 276, 276, 3), dtype=float32)

    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0],
                 tf.stack([-1, height - 20, width - 20, -1]))  # Tensor("Slice:0", shape=(?, ?, ?, ?), dtype=float32)
    return y

