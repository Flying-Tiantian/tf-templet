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
from input_generator import file_reader
from hyper_params import hyparams as param
from models import alexnet

FLAGS = None


def train():
    data_reader = file_reader(fake=True, fake_class_num=1001)
    hyparams = param()
    hyparams.data_info(data_reader)

    if FLAGS.init_lr == 0:
        FLAGS.init_lr = hyparams.init_lr
    else:
        hyparams.init_lr = FLAGS.init_lr
    if FLAGS.max_steps == 0:
        FLAGS.max_steps = hyparams.max_steps
    else:
        hyparams.max_steps = FLAGS.max_steps

    model = alexnet.alexnet_model(hyparams)

    FLAGS.train_dir = os.path.join(FLAGS.train_dir, model.get_name())
    
    if tf.gfile.Exists(FLAGS.train_dir):
        pass
        # tf.gfile.DeleteRecursively(FLAGS.train_dir)
    else:
        tf.gfile.MakeDirs(FLAGS.train_dir)

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        acc = tf.Variable(tf.constant(1.0, dtype=tf.float32), trainable=False, name='acc')
        tf.summary.scalar('accuracy', acc)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_imgs, train_labels = data_reader.distorted_inputs(
            hyparams.batch_size, model.get_input_size())
        train_data = {
            'images': train_imgs,
            'labels': train_labels
        }
        train_op, loss_op = model.train(train_data)

        test_imgs, test_labels = data_reader.inputs(
            hyparams.test_batch_size, model.get_input_size())
        test_data = {
            'images': test_imgs,
            'labels': test_labels
        }
        top_k_op, logits, labels = model.test(test_data)

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            # Start the queue runners.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print('Init all variables...')
            sess.run(tf.global_variables_initializer())
            print('Complete.')

            gstepvalue = model.restore(sess, FLAGS.train_dir)
            if gstepvalue != -1:
                sess.run(tf.assign(global_step, tf.constant(
                    gstepvalue, dtype=tf.int64)))
                print('Recover from global step %d.' % gstepvalue)

            for i in range(hyparams.max_steps):
                _, loss, step = sess.run([train_op, loss_op, global_step])
                if step % hyparams.save_checkpoint_interval == 0:
                    model.save(sess, FLAGS.train_dir + '/model.ckpt')
                if step % hyparams.save_summary_interval == 0:
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, global_step=step)
                if step % hyparams.test_interval == 0:
                    epoch = int(step / hyparams.test_interval)
                    true_count = 0  # Counts the number of correct predictions.
                    total_sample_count = hyparams.test_steps * hyparams.test_batch_size
                    for i in range(hyparams.test_steps):
                        predictions, score, ground_truth = sess.run([top_k_op, logits, labels])
                        true_count += np.sum(predictions)
                        result = np.argmax(score, axis=1)
                        # print(result)
                        # print(ground_truth)
                    # Compute precision @ 1.
                    precision = true_count / total_sample_count
                    sess.run(tf.assign(acc, tf.constant(precision, dtype=tf.float32)))
                    print('%s: epoch %d, loss %.2f, precision @ 1 = %.3f' %
                          (datetime.datetime.now(), epoch, loss, precision))

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./data/train/',
        help='Directory where to write event logs and checkpoint.')
    parser.add_argument(
        '--init_lr',
        type=float,
        default=0,
        help='The initial learning rate.')
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=1,
        help='The number of gpu to use.')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=0,
        help='Number of batches to run.')
    parser.add_argument(
        '--log_device_placement',
        type=bool,
        default=False,
        help='Whether to log device placement.')

    FLAGS, unparsed = parser.parse_known_args()
    main(argv=[sys.argv[0]] + unparsed)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
