#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:48:14 2018

@author: hellcat
"""

import main
import tensorflow as tf
from general_net import net
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

opt = main.Config()
slim = tf.contrib.slim
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

text_img = './000000000036.jpg'
model_path = "./logs/model"

img_raw = tf.gfile.FastGFile(text_img,'rb').read()
img = tf.image.decode_jpeg(img_raw)
if img.dtype != tf.float32:
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
generated = net(tf.expand_dims(img, axis=0), training=False)

with tf.Session() as sess:
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        tf.logging.info("Success to read {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        tf.logging.info("Failed to find a checkpoint")
    img_gen = sess.run(generated)[0]
    img_gen = img_gen.astype(int)
    plt.imshow(img_gen)
    img_gen = Image.fromarray(img_gen.astype(np.uint8))
    img_gen.save('./output_img.png')