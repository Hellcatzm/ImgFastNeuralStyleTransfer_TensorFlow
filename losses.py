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
from train import gram


def content_loss(endpoints_dict, content_layers):
    c_loss = 0
    c_loss_summary = {}
    for layer in content_layers:
        # 将生成图和原图输出分离
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        layer_content_loss = tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)
        c_loss_summary[layer] = layer_content_loss
        c_loss += layer_content_loss
    return c_loss, c_loss_summary
'''
tf.size:
This operation returns an integer representing the number of elements in
`input`.
For example:
```python
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.size(t)  # 12
```
'''


def style_loss(endpoints_dict, style_features_t, style_layers):
    s_loss = 0
    s_loss_summary = {}
    # 计算好gram的style特征，需要计算style的层
    for style_gram, layer in zip(style_features_t, style_layers):
        # 获取生成网络的style节点
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        # 计算l2 loss，一个batch的图片对目标特征求解损失
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        s_loss_summary[layer] = layer_style_loss
        s_loss += layer_style_loss
    return s_loss, s_loss_summary


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    # 逐像素做差
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1]))\
        - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1]))\
        - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    # 合并x,y两方向的l2 loss
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss