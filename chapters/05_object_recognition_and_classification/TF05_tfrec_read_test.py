# -*- coding: utf-8 -*-
import os
import numpy as np
import struct
from PIL import Image
import tensorflow as tf
import io

'''
验证tfrecord文件生成的是否正确
'''

# Dog Files
cwd = "c:\\work\\070_DL\\data\\StanfordDog\\output_test\\"
filename_queue = tf.train.string_input_producer(["c:\\work\\070_DL\\data\\StanfordDog\\output\\testing-images\\testing-image-0.tfrecords"]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.string),
                                       'file_name': tf.FixedLenFeature([], tf.string),
                                       'image' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
# 1，保存为raw image时，这样读出
'''
image = tf.decode_raw(features['image'], tf.uint8)
# [151, 250, 3] for color image
image = tf.reshape(image, [151, 250])   # height, width, 和PIL.Image的定义不同
'''
# 2，保存为encode image时，直接读出
image = features['image']
label = features['label']
fname = features['file_name']

with tf.Session() as sess: #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(200):
        example, file_name, l = sess.run([image,fname, label])#在会话中取出image和label

        # 1，raw image：
        '''
        # 'RGB' for color image, 'L' for gray
        # img=Image.fromarray(example, 'L')#这里Image是之前提到的

        # *.decode('utf-8'),  bytes to string
        img.save(cwd+'Label_'+ str(l)+'_' + str(i) + '_' + file_name.decode('utf-8') + '.jpg')#存下图片
        #print(example, l)
        '''

        # 2，encode image，直接保存，filename中已经包含了后缀名
        f = open(cwd + 'Label_' + str(l) + '_' + str(i) + '_' + file_name.decode('utf-8'), 'wb')
        f.write(example)
        # 这时，也可以使用PIL Image decode image
        # encoded_img = io.BytesIO(example)
        # img=Image.open(encoded_img)


    coord.request_stop()
    coord.join(threads)
