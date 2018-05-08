快速风格迁移实践
=============
## 资源
[vgg16预训练模型](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)<br>
[COCO2017训练数据](http://images.cocodataset.org/zips/train2017.zip)<br>
[COCO2017训练标签](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)<br>
[COCO2017验证数据](http://images.cocodataset.org/zips/val2017.zip)<br>
[COCO2017验证标签](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)<br>
[COCO2017测试数据](http://images.cocodataset.org/zips/test2017.zip)<br>
[COCO2017测试标签](http://images.cocodataset.org/annotations/image_info_test2017.zip)<br>
本次实验仅仅是用来训练数据(不含标签)，后面的5个链接是为了数据完整性给出的，针对本次实验不需下载。

## 结构
![图](https://github.com/Hellcatzm/FastNeuralStyleTransfer_tensorflow/blob/master/%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E7%A4%BA%E6%84%8F%E5%9B%BE.png "项目结构可视化")

## 日志
#### 18.5.7
##### 1、改正`main.py`中"tf.local_variables"为"tf.global_variables"
之前测试时修改过去忘记改回来了，这导致局部变量epoch被保存，导致之后的训练如果继承了前面训练过得模型会继承epoch信息报告到达预定轮数直接退出，另由于没有载入模型参数(均为global_variables)会导致训练不收敛<br>
##### 2、改正`main.py`中两处"vgg16"为"vgg_16"
这个疏忽导致"variables_to_train"和"variables_to_restore"收集变量异常，进而导致网络不收敛<br>
##### 3、改进训练循环中的保存图片机制
由原来如下的常规图像处理，修改为summary记录机制。
```Python
if not os.path.exists('./保存图像'):
    os.makedirs('./保存图像')
    img_raw = sess.run(image_batch)[0]
    img_gen = sess.run(generated)[0]
    try:
        img_raw = Image.fromarray(np.uint8(img_raw))
        img_raw.save('./保存图像/{}_.png'.format(step))
        img_gen = Image.fromarray(np.uint8(img_gen))
        img_gen.save('./保存图像/{}.png'.format(step))
    except BaseException as e:
        tf.logging.info(e)
```
由于sess一次原始数据，在使用队列维护数据机制下，就会消耗一个batch的数据，频繁记录输出图片会导致参与训练的数据减少严重，所以在小数据测试时我发现实际训练step少于理论上计算值，就是因为很多batch的数据没有被训练被用于生成图像了，如果将输出图像和train_op集成后一起run也可以解决，不过增加了很多冗余输出，毕竟不是每个step都要可视化的。
```Python
_, loss_t, step, img_raw, img_gen = sess.run([train_op, loss, global_step, image_batch, generated])
```
