# Author : hellcat
# Time   : 18-5-3

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
"""

import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
import losses
from nets import vgg
from general_net import net

slim = tf.contrib.slim
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Config(object):

    data_root = 'train2017'  # 数据集存放路径：train2017/a.jpg
    image_size = 256  # vgg16: To use in classification mode, resize input to 224x224.
    batch_size = 4
    epoches = 2  # 训练epoch

    model_path = "pretrained/vgg_16.ckpt"  # 预训练模型的路径
    exclude_scopes = "vgg_16/fc"

    style_layers = ("vgg_16/conv1/conv1_2",
                    "vgg_16/conv2/conv2_2",
                    "vgg_16/conv3/conv3_3",
                    "vgg_16/conv4/conv4_3")  # 风格学习层
    style_path = "img/mosaic.jpg"  # 风格图片存放路径

    content_layers = ["vgg_16/conv3/conv3_3", ]

    lr = 1e-3  # 学习率
    content_weight = 1  # content_loss 的权重
    style_weight = 100  # style_loss的权重
    tv_weight = 0.0


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
    image = utils.img_proprocess(image_row, opt.image_size)
    image_batch = tf.train.batch([image], opt.batch_size, dynamic_pad=True)

    '''生成式网络生成数据'''
    generated = net(image_batch, training=True)
    generated = tf.image.resize_bilinear(generated, [opt.image_size, opt.image_size], align_corners=False)
    generated.set_shape([opt.batch_size, opt.image_size, opt.image_size, 3])
    # unstack将指定维度拆分为1后降维，split随意指定拆分后维度值且不会自动降维
    # processed_generated = tf.stack([utils.img_proprocess(tf.squeeze(img, axis=0), opt.image_size)
    #                                 for img in tf.split(generated, num_or_size_splits=opt.batch_size, axis=0)])
    processed_generated = tf.stack([utils.img_proprocess(img, opt.image_size) for img in tf.unstack(generated, axis=0)])

    '''数据流经损失网络_VGG'''
    # 一次送入数据量为2×batch_size：[原始batch经生成式网络生成的数据 + 原始batch]
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=0.0)):  # 调用
        _, endpoint = vgg.vgg_16(tf.concat([processed_generated, image_batch], 0),
                                 num_classes=1, is_training=False, spatial_squeeze=False)
    tf.logging.info('Loss network layers(You can define them in "content_layers" and "style_layers"):')
    for key in endpoint:
        tf.logging.info(key)

    '''损失函数构建'''
    style_gram = utils.get_style_feature(opt.style_path,
                                         opt.image_size,
                                         opt.style_layers,
                                         opt.model_path,
                                         opt.exclude_scopes)
    content_loss, content_loss_summary = losses.content_loss(endpoint, opt.content_layers)
    style_loss, style_loss_summary = losses.style_loss(endpoint, style_gram, opt.style_layers)
    tv_loss = losses.total_variation_loss(generated)  # use the unprocessed image, 我们想要的图像也是这个
    loss = opt.style_weight * style_loss + opt.content_weight * content_loss + opt.tv_weight * tv_loss

    '''优化器构建'''
    # 优化器维护非vgg16的可训练变量
    variable_to_train = []
    for variable in tf.trainable_variables():
        if not (variable.name.startswith("vgg_16")):  # "vgg16"
            variable_to_train.append(variable)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)

    '''存储器构建'''
    # 存储器保存非vgg16的全局变量

    # tf.global_variables()：返回全局变量。
    # 全局变量是分布式环境中跨计算机共享的变量。该Variable()构造函数或get_variable()
    # 自动将新变量添加到图形集合：GraphKeys.GLOBAL_VARIABLES。这个方便函数返回该集合的内容。
    # 全局变量的替代方法是局部变量。参考：tf.local_variables
    variables_to_restore = []  # 比trainable多出的主要是用于bp的变量
    for variable in tf.global_variables():
        if not (variable.name.startswith("vgg_16")):  # "vgg16"
            variables_to_restore.append(variable)

    saver = tf.train.Saver(var_list=variables_to_restore, write_version=tf.train.SaverDef.V2)

    print(variables_to_restore)

    with open('train_v.txt', 'w') as f:
        for s in variable_to_train:
            f.write(s.name + '\n')
    with open('restore_v.txt', 'w') as f:
        for s in variables_to_restore:
            f.write(s.name + '\n')
    '''训练'''
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        # vgg网络预训练参数载入
        param_init_fn = utils.param_load_fn(opt.model_path, opt.exclude_scopes)
        param_init_fn(sess)

        # 由于使用saver，故载入变量不包含vgg16相关变量
        model_path = "./logs/model"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt:
            tf.logging.info("Success to read {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.logging.info("Failed to find a checkpoint")

        coord = tf.train.Coordinator()  # 线程控制器
        threads = tf.train.start_queue_runners(coord=coord)  # 启动队列
        start_time = time.time()  # 计时开始
        try:
            while not coord.should_stop():
                _, loss_t, step = sess.run([train_op, loss, global_step])
                elapsed_time = time.time() - start_time
                start_time = time.time()

                if step % 10 == 0:
                    tf.logging.info('step: {0:d}, total Loss {1:.2f}, secs/step: {2:.3f}'.
                                    format(step, loss_t, elapsed_time))
                    # if not os.path.exists('./保存图像'):
                    #     os.makedirs('./保存图像')
                    # img_0 = sess.run(image_batch)[0]
                    # img = sess.run(generated)[0]
                    # try:
                    #     img_0 = Image.fromarray(np.uint8(img_0))
                    #     img_0.save('./保存图像/{}_.png'.format(step))
                    #     img = Image.fromarray(np.uint8(img))
                    #     img.save('./保存图像/{}.png'.format(step))
                    # except BaseException as e:
                    #     tf.logging.info(e)

                if step % 1000 == 0:
                    saver.save(sess, os.path.join(model_path, 'fast_style_model'), global_step=step)

        except tf.errors.OutOfRangeError:
            saver.save(sess, os.path.join(model_path, 'fast_style_model'))
            tf.logging.info('Epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

    '''测试部分'''
    # print('style_gram:', [f.shape for f in style_gram])
    # with tf.Session(config=config) as sess:
    #     sess.run(tf.group(tf.global_variables_initializer(),
    #                       tf.local_variables_initializer()))
    #     coord = tf.train.Coordinator()  # 线程控制器
    #     threads = tf.train.start_queue_runners(coord=coord)  # 启动队列
    #     try:
    #         while not coord.should_stop():
    #             row = sess.run(image_row)
    #             img = sess.run(image)
    #             # img_re = utils.image_rebuild(img[0])
    #             # print(img[0])
    #             # plt.subplot(311)
    #             # plt.imshow(row)
    #             # plt.subplot(312)
    #             # plt.imshow(img[0])
    #             # plt.subplot(313)
    #             # plt.imshow(img_re)
    #             # plt.show()
    #     except tf.errors.OutOfRangeError:
    #         tf.logging.info('Done training -- epoch limit reached')
    #     finally:
    #         coord.request_stop()
    #     coord.join(threads)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    train()

