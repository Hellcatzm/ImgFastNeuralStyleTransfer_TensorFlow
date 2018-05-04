# Author : hellcat
# Time   : 18-5-3

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
import os
import numpy as np
from nets import vgg
import tensorflow as tf

slim = tf.contrib.slim
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

IMAGENET_MEAN = [123.68, 116.78, 103.94]
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224,  0.225]


def img_proprocess(image_row,
                   image_size=224,
                   process=True):
    """
    预处理图片
    :param image_row: 三维的图片张量
    :param image_size: 处理后图片尺寸
    :param process: 是否进行均值方差操作
    :return: 预处理之后的四维图片张量
    """
    # 图片维度和尺寸调整
    image = tf.expand_dims(image_row, 0)
    image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
    image.set_shape([1, image_size, image_size, 3])

    if process:
        num_channels = image.get_shape().as_list()[-1]
        channels = tf.split(image, num_or_size_splits=num_channels, axis=3)
        for i in range(num_channels):
            channels[i] -= IMAGENET_MEAN[i]
            # channels[i] = (channels[i]/255 - IMAGENET_MEAN[i])/IMAGENET_STD[i]
        image = tf.concat(channels, axis=3)
    return image


# def image_rebuild(image):
#     image_re = np.array(image)
#     print(image_re.shape)
#     for i in range(3):
#         image_re[:, :, i] = image[:, :, i]*255#(image[:, :, i] + IMAGENET_MEAN[i])*255
#     return image_re


def param_load_fn(model_path="pretrained/vgg_16.ckpt",
                  exclude_scopes="vgg_16/fc"):
    """
    自动获取图网络结构并载入预训练模型
    :param model_path: 模型文件位置，注意是所ckpt文件
    :param exclude_scopes: 不予加载的节点的名称字符串："a,b"
    :return: 
    """
    tf.logging.info('Use pretrained model %s' % model_path)

    exclusions = []
    if exclude_scopes:  # "vgg_16/fc"
        exclusions = [scope.strip()  # 舍弃的节点的名称头集合
                      for scope in exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():  # 获取图中的变量
        excluded = False
        for exclusion in exclusions:  # 获取舍弃头字符
            # 对于所有的舍弃头，判断当前变量对应节点是否从属于其
            if var.op.name.startswith(exclusion):  # 变量对应节点
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)  # 添加进重载变量集合
    # 此函数调用时需要一个参数，sess，对应的模型只需要路径即可
    return slim.assign_from_checkpoint_fn(  # 从checkpoint中读取列表中的相应变量参数值的函数
        model_path,
        variables_to_restore,
        ignore_missing_vars=True)


def gram(layer):
    """
    格拉姆矩阵计算
    """
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    # gram矩阵求解，transpose_a=True意为乘法前对a(第一个乘数)进行转置
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)  # 求均值，原来没有
    return grams


def get_style_feature(style_path="img/mosaic.jpg",
                      image_size=224,
                      style_layers=("vgg_16/conv1/conv1_2",
                                    "vgg_16/conv2/conv2_2",
                                    "vgg_16/conv3/conv3_3",
                                    "vgg_16/conv4/conv4_3"),
                      model_path="pretrained/vgg_16.ckpt",
                      exclude_scopes="vgg_16/fc"):
    """
    使用预训练的vgg网络，获取并return style层对应的特征
    :param style_path: 
    :param image_size: 
    :param style_layers:
    :param model_path:
    :param exclude_scopes:
    :return: 由于使用单独的Graph，返回值为Array而非Tensor
    """
    with tf.Graph().as_default():

        # 风格图片载入
        img_bytes = tf.read_file(style_path)
        if style_path.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = img_proprocess(image, image_size)

        # vgg网络节点接口生成
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=0.0)):  # 调用
            _, endpoint = vgg.vgg_16(image, num_classes=1, is_training=False, spatial_squeeze=False)
        features = []
        for layer in style_layers:
            feature = endpoint[layer]
            feature = tf.squeeze(gram(feature), [0])  # 张量生成gram矩阵，并舍弃第零维(batch)
            features.append(feature)  # 将张量对象添加进features

        with tf.Session(config=config) as sess:
            # vgg网络预训练参数载入
            param_init_fn = param_load_fn(model_path, exclude_scopes)
            param_init_fn(sess)

            # Make sure the 'generated' directory is exists.
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            style_name = style_path.split('/')[-1].split('.')[0]
            save_file = 'generated/target_style_' + style_name + '.jpg'
            with open(save_file, 'wb') as f:

                # 各通道均值归零还原
                num_channels = image.get_shape().as_list()[-1]
                channels = tf.split(image, num_or_size_splits=num_channels, axis=3)
                for i in range(num_channels):
                    channels[i] += IMAGENET_MEAN[i]
                target_image = tf.concat(channels, 3)

                value = tf.image.encode_jpeg(tf.cast(target_image[0], tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            # Return the features those layers are use for measuring style loss.
            return sess.run(features)  # 返回特征

