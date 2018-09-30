import os
import sys
import tarfile

import pandas as pd
from six.moves import urllib

import tensorflow as tf


DATA_DIR = os.path.join('data', 'imagenet_data')

TRAIN_LIST = os.path.join(DATA_DIR, 'train_list.txt')
VAL_LIST = os.path.join(DATA_DIR, 'val_list.txt')

def creat_list(data_dir):
    def _make_train_val_list(data_dir):
        classes = sorted(os.listdir(data_dir))

        data = []
        class_num = len(classes)

        for one_class, index in zip(classes, range(class_num)):
            class_path = os.path.join(data_dir, one_class)
            img_files = sorted(os.listdir(class_path))
            img_files = [os.path.join(class_path, filename) for filename in img_files]
            example_list = {'path': img_files, 'label': [index] * len(img_files)}
            data.append(pd.DataFrame(data=example_list))

        return pd.concat(data), class_num

    train_data_dir = os.path.join(data_dir, 'train')
    train_list, train_class_num = _make_train_val_list(train_data_dir)
    train_list = train_list.sample(frac=1)

    val_data_dir = os.path.join(data_dir, 'val')
    val_list, val_class_num = _make_train_val_list(val_data_dir)
    val_list = val_list.sample(frac=1)

    if train_class_num != val_class_num:
        raise ValueError('Train and validate dataset has different class number!')

    return train_list.values.tolist(), val_list.values.tolist(), train_class_num



class file_reader:
    def __init__(self, fake=False, fake_class_num=None):
        self._fake = fake
        self.num_preprocess_threads = 4
        self.min_queue_examples = 100
        self._train_list, self._val_list, self._class_num = creat_list(DATA_DIR)
        if self._fake:
            self._class_num = fake_class_num
        pass

    def class_num(self):
        return self._class_num

    def example_num_for_train(self):
        return len(self._train_list)

    def example_num_for_eval(self):
        return len(self._val_list)

    def _name_queue(self, name_list, label_list):
        example = tf.train.slice_input_producer([name_list, label_list], shuffle=False)

        return example[0], example[1]

    def _file_reader(self, filename):
        value = tf.read_file(filename)
        '''reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)'''
        
        image = tf.image.decode_jpeg(value, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        

        return image

    def _img_preprocess(self, image, size):
        shape = tf.shape(image)
        dims = tf.unstack(shape)
        cent_size = tf.minimum(dims[0], dims[1])
        square_img = tf.image.resize_image_with_crop_or_pad(image, cent_size, cent_size)
        resized = tf.image.resize_images(square_img, (size, size))
        standardized = tf.image.per_image_standardization(resized)
        standardized.set_shape([size, size, 3])

        return standardized

    def _img_expand_preprocess(self, image, size):
        shape = tf.shape(image)
        dims = tf.unstack(shape)
        cent_size = tf.minimum(dims[0], dims[1])
        cent_size = tf.cast(tf.multiply(tf.cast(cent_size, tf.float32), 0.8), tf.int32)

        distorted_image = tf.random_crop(image, [cent_size, cent_size, 3])

        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_flip_up_down(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        distorted_image = tf.image.random_hue(distorted_image, 0.2)
        distorted_image = tf.image.random_saturation(distorted_image, 0.2, 0.8)

        resized = tf.image.resize_images(distorted_image, (size, size))

        # Subtract off the mean and divide by the variance of the pixels.
        standardized = tf.image.per_image_standardization(resized)

        # Set the shapes of tensors.
        standardized.set_shape([size, size, 3])

        return standardized

    def _make_label(self, label):
        label = tf.cast(label, tf.int32)

        return label

    def _gen_batch(self, image, label, batch_size, shuffle):
        if shuffle:
            image_batch, label_batch = tf.train.shuffle_batch(
                    [image, label],
                    batch_size=batch_size,
                    num_threads=self.num_preprocess_threads,
                    capacity=self.min_queue_examples + 3 * batch_size,
                    min_after_dequeue=self.min_queue_examples)
        else:
            image_batch, label_batch = tf.train.batch(
                    [image, label],
                    batch_size=batch_size,
                    num_threads=self.num_preprocess_threads,
                    capacity=self.min_queue_examples + 3 * batch_size)

        return image_batch, tf.reshape(label_batch, [batch_size])

    def inputs(self, batch_size, image_size, distorted=False):
        example_list = self._val_list
        paths = [element[0] if isinstance(element[0], str) else element[1] for element in example_list]
        labels = [element[0] if isinstance(element[0], int) else element[1] for element in example_list]

        with tf.name_scope('input_generator') as scope:
            filename, label = self._name_queue(paths, labels)
            if distorted:
                image = self._img_expand_preprocess(self._file_reader(filename), image_size)
            else:
                image = self._img_preprocess(self._file_reader(filename), image_size)
            label = self._make_label(label)
            image_batch, label_batch = self._gen_batch(image, label, batch_size, False)

        return image_batch, label_batch

    def test(self, batch_size=1):
        example_list = self._val_list
        paths = [element[0] if isinstance(element[0], str) else element[1] for element in example_list]
        labels = [element[0] if isinstance(element[0], int) else element[1] for element in example_list]
        filename, label = self._name_queue(paths, labels)
        name_batch, label_batch = self._gen_batch(filename, label, batch_size, True)

        return name_batch, label_batch

    def distorted_inputs(self, batch_size, image_size):
        return self.inputs(batch_size, image_size, distorted=True)


if __name__ == '__main__':
    reader = file_reader()

    name_batch, label_batch = reader.test()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        label_dict = {}
        for i in range(100):
            name_list, label_list = sess.run([name_batch, label_batch])
            for name, label in zip(name_list, label_list):
                key = str(os.path.split(name)[-2], encoding='utf-8').split('\\')[-1]
                if key in label_dict:
                    label_dict[key] += [label]
                else:
                    label_dict[key] = [label]
                
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        pass

