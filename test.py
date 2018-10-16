# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to train pruned CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py when target sparsity in
cifar10_pruning_spec.pbtxt is set to zero

Results:
Sparsity | Accuracy after 150K steps
-------- | -------------------------
0%       | 86%
50%      | 86%
75%      | TODO(suyoggupta)
90%      | TODO(suyoggupta)
95%      | 77%

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import sys
import os
import time

import numpy as np
import tensorflow as tf

from hyper_params import hyparams as param
from models import squeezenet
from input_generator import file_reader

FLAGS = None


def test():
    data_reader = file_reader(fake=True, fake_class_num=1001)
    hyparams = param()
    hyparams.data_info(data_reader)

    model = squeezenet.squeezenet_model(hyparams)

    with tf.Graph().as_default():
        if FLAGS.use_train_data:
            test_imgs, test_labels = data_reader.distorted_inputs(
                hyparams.test_batch_size, model.get_input_size())
        else:
            test_imgs, test_labels = data_reader.inputs(
                hyparams.test_batch_size, model.get_input_size())
        test_data = {
            'images': test_imgs,
            'labels': test_labels
        }
        top_k_op = model.test(test_data)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print('Init all variables...')
            sess.run(tf.global_variables_initializer())
            print('Complete.')

            gstepvalue = model.restore(sess, FLAGS.restore_dir)
            if gstepvalue == -1:
                pass
                # raise ValueError("can not find checkpoint!")

            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = 0
            print('%s:start' % datetime.datetime.now())
            for i in range(hyparams.test_steps):
                predictions = sess.run(top_k_op)
                total_sample_count += len(predictions)
                true_count += np.sum(predictions)
                print('%s:step %d' % (datetime.datetime.now(), i))
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s:precision @ 1 = %.3f' % (datetime.datetime.now(), precision))

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument
    test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_dir',
        type=str,
        default='./data/saved_model/',
        help='Directory where to write event logs and checkpoint.')
    parser.add_argument(
        '--use_train_data',
        type=bool,
        default=False,
        help='Whether to test train acc.')

    FLAGS, unparsed = parser.parse_known_args()
    main(argv=[sys.argv[0]] + unparsed)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
