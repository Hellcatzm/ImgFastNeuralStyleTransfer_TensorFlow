# Author : hellcat
# Time   : 18-5-3

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
from nets import vgg
from general_net import net

slim = tf.contrib.slim
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
means = [_R_MEAN, _G_MEAN, _B_MEAN]
# std = [0.229, 0.224, 0.225]


class Config(object):
    # vgg16: To use in classification mode, resize input to 224x224.
    image_size = 224
    batch_size = 8
    data_root = 'train2017'  # 数据集存放路径：train2017/a.jpg
    num_workers = 4  # 多线程加载数据
    use_gpu = True  # 使用GPU

    style_path = 'style.jpg'  # 风格图片存放路径
    lr = 1e-3  # 学习率

    plot_every = 10  # 每10个batch可视化一次

    epoches = 2  # 训练epoch

    content_weight = 1e5  # content_loss 的权重
    style_weight = 1e10  # style_loss的权重

    model_path = None  # 预训练模型的路径

    content_path = 'input.png'  # 需要进行分割迁移的图片
    result_path = 'output.png'  # 风格迁移结果的保存路径


def train(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    '''数据加载'''
    # 读取文件名
    filenames = [os.path.join(opt.data_root, f)
                 for f in os.listdir(opt.data_root)
                 if os.path.isfile(os.path.join(opt.data_root, f))]
    # 判断文件格式，png为True，jpeg为False
    png = filenames[0].lower().endswith('png')  # If first file is a png, assume they all are
    # 维持文件名队列
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True, num_epochs=opt.epoches)
    # 初始化阅读器
    reader = tf.WholeFileReader()
    # 返回tuple，是key-value对
    _, img_bytes = reader.read(filename_queue)
    # 图片格式解码
    image_row = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    # 预处理
    image = utils.img_proprocess(image_row, opt.image_size, means)
    image_batch = tf.train.batch([tf.squeeze(image)], opt.batch_size, dynamic_pad=True)

    '''生成式网络生成数据'''
    generated = net(image_batch, training=True)
    generated = tf.image.resize_bilinear(generated, [opt.image_size, opt.image_size], align_corners=False)
    generated.set_shape([opt.batch_size, opt.image_size, opt.image_size, 3])

    '''载入损失网络_VGG'''
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=0.0)):  # 调用
        _, endpoint = vgg.vgg_16(tf.concat([generated, image_batch], 0),
                                 num_classes=1, is_training=False, spatial_squeeze=False)
    tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
    for key in endpoint:
        tf.logging.info(key)

    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        coord = tf.train.Coordinator()  # 线程控制器
        threads = tf.train.start_queue_runners(coord=coord)  # 启动队列
        try:
            while not coord.should_stop():
                row = sess.run(image_row)
                img = sess.run(image)
                print(row.shape, img[0].shape)
                # plt.subplot(211)
                # plt.imshow(row)
                # plt.subplot(212)
                plt.imshow(img[0])
                plt.show()
        except tf.errors.OutOfRangeError:
            tf.logging.info('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    style_gram = utils.get_style_feature()
    train()

