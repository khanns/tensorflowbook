# -*- coding: utf-8 -*-
import os
import numpy as np
import struct
import PIL.Image

import tensorflow as tf
import glob

# Dog Files

from itertools import groupby
from collections import defaultdict

def write_records_file(dataset, record_location):
    """
    Fill a TFRecords file with the images found in `dataset` and include their category.

    Parameters
    ----------
    dataset : dict(list)
      Dictionary with each key being a label for the list of image filenames of its value.
    record_location : str
      Location to store the TFRecord output.

    这个函数运行非常慢，并且会导致内存溢出。感觉是

    """
    writer = None

    # Enumerating the dataset because the current index is used to breakup the files if they get over 100
    # images to avoid a slowdown in writing.
    current_index = 0
    for breed, images_filenames in dataset.items():
        print(breed)
        for image_filename in images_filenames:
            if current_index % 1000 == 0:
                if writer:
                    writer.close()

                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1
            print(current_index)

            image_file = tf.read_file(image_filename)

            # In ImageNet dogs, there are a few images which TensorFlow doesn't recognize as JPEGs. This
            # try/catch will ignore those images.
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue

            # Converting to grayscale saves processing and memory but isn't required.
            # grayscale_image = tf.image.rgb_to_grayscale(image)
            
            # resized_image = tf.image.resize_images(grayscale_image, (250, 151))

            # tf.cast is used here because the resized images are floats but haven't been converted into
            # image floats where an RGB value is between [0,1).
            image_bytes = sess.run(tf.cast(image, tf.uint8)).tobytes()

            # Instead of using the label as a string, it'd be more efficient to turn it into either an
            # integer index or a one-hot encoded rank one tensor.
            # https://en.wikipedia.org/wiki/One-hot
            image_label = breed.encode("utf-8")

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            # writer.write(example.SerializeToString())
            writer.write(example.SerializeToString())
    writer.close()
    
    
sess = tf.InteractiveSession()

image_filenames = glob.glob("..\\data\\StanfordDog\\imagenet-dogs\\n02*\\*.jpg")
image_filenames[0:2]

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# Split up the filename into its breed and corresponding filename. The breed is found by taking the directory name
image_filename_with_breed = map(lambda filename: (filename.split("\\")[-2], filename), image_filenames)


# Group each image by the breed which is the 0th element in the tuple returned above
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    # Enumerate each breed's image and send ~20% of the images to a testing set
    print(dog_breed)
    for i, breed_image in enumerate(breed_images):
        # print(i)
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])

    # Check that each breed includes at least 18% of the images for testing
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])
    print(breed_training_count, breed_testing_count)
    assert round(breed_testing_count / (breed_training_count + breed_testing_count), 2) > 0.18, "Not enough testing images."
'''
testing_dataset是collections.defaultdict类型，有keys()和values()方法。调用这两个方法以后，得到的是dict_values类型，可以使用list函数获取具体的值，并显示出来
type(testing_dataset)
Out[2]: collections.defaultdict
type(testing_dataset.values())
Out[4]: dict_values

len(testing_dataset.values())
Out[5]: 120
len(testing_dataset.keys())
Out[8]: 120

list(testing_dataset.keys())[0]
Out[11]: 'n02085620-Chihuahua'
list(testing_dataset.keys())[2]
Out[12]: 'n02085936-Maltese_dog'

'''
write_records_file(testing_dataset, "../data/StanfordDog/output/testing-images/testing-image")
# write_records_file(training_dataset, "../data/StanfordDog/output/training-images/training-image")

