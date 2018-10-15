import os
import tensorflow as tf
import abc

from .model_util import *

class base_model(object):
    """Abstract meta class for all models.
    Implement save and restore function, and single/multi gpu forward pass.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyparams):
        self._hyparams = hyparams
        self._name = None
        self._v_scope = None
        self._saver = None
        self._input_size = None

    @abc.abstractmethod
    def _forward_pass(self, input_layer, trainable=True):
        pass

    def forward_pass(self, input_layer, trainable=True):
        if self._v_scope:
            with tf.variable_scope(self._v_scope):
                r = self._forward_pass(input_layer, trainable)
        else:
            r = self._forward_pass(input_layer, trainable)
        
        return r

    """def forward_pass(self, input_layer, gpu_num=1, trainable=True):
        def _f(self, input_layer, gpu_num, trainable):
            if gpu_num == 1:
                return [self._forward_pass(self, input_layer, trainable)]
            final_layers = []
            for i in range(gpu_num):
                with tf.name_scope('tower_%d' % i):
                    with tf.device('gpu:%d' % i):
                        final_layers.append(self._forward_pass(self, input_layer, trainable))

            return final_layers

        if self._name:
            with tf.variable_scope(self._name):
                final_layers = _f(self, input_layer, gpu_num, trainable)
        else:
            final_layers = _f(self, input_layer, gpu_num, trainable)

        return final_layers"""

    @abc.abstractmethod
    def train(self, train_data):
        pass

    @abc.abstractmethod
    def test(self, test_data):
        pass

    def get_name(self):
        return self._name

    def get_input_size(self):
        return self._input_size

    def _get_saver(self):
        if self._saver is None:
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self._v_scope)
            self._saver = tf.train.Saver(variables, max_to_keep=self._hyparams.max_to_keep)

        return self._saver

    def save(self, sess, save_path):
        saver = self._get_saver()

        return saver.save(sess, save_path, tf.train.get_global_step())

    def restore(self, sess, checkpoint_dir):
        # pylint: disable=no-member
        if os.path.exists(checkpoint_dir):
            if os.path.isdir(checkpoint_dir):
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    checkpoint_path = ckpt.model_checkpoint_path
                else:
                    checkpoint_path = os.path.join(checkpoint_dir, self._name + '.ckpt')
                    if not os.path.exists(checkpoint_path):
                        return -1
            else:
                checkpoint_path = checkpoint_dir
            saver = self._get_saver()
            # Restores from checkpoint
            print('Restore from %s ...' % checkpoint_path)
            saver.restore(sess, checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = checkpoint_path.split(
                '/')[-1].split('-')[-1]

            try:
                global_step = int(global_step)
            except ValueError:
                global_step = 0
            return global_step

        else:
            print('No checkpoint file found')
            return -1
        # pylint: enable=no-member

class classification_model(base_model):
    """Abstract meta class for classifier model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyparams):
        base_model.__init__(self, hyparams)

    @abc.abstractmethod
    def _forward_pass(self, input_layer, trainable=True):
        pass

    def _loss(self, logits, labels):
        loss_dict = {}

        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)

        acc_loss = tf.add_n(tf.get_collection(
            tf.GraphKeys.LOSSES), name='accuracy_loss')
        tf.summary.scalar('accuracy_loss', acc_loss)
        loss_dict['accuracy_loss'] = acc_loss

        l2_loss = tf.add_n(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')
        tf.summary.scalar('l2_loss', l2_loss)
        loss_dict['l2_loss'] = l2_loss

        total_loss = tf.add(acc_loss, l2_loss, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)
        loss_dict['total_loss'] = total_loss

        return loss_dict

    def train(self, train_data):
        with tf.name_scope('train'):
            logits = self.forward_pass(train_data['images'])
            tf.summary.image('train_img', train_data['images'])
            loss_dict = self._loss(logits, train_data['labels'])

            global_step = tf.train.get_global_step()

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(
                    self._hyparams.init_lr,
                    global_step,
                    self._hyparams.decay_steps,
                    self._hyparams.lr_decay,
                    staircase=True)
            tf.summary.scalar('learning_rate', lr)
            
            opt = tf.train.AdamOptimizer(learning_rate=lr)
            train_op = opt.minimize(loss_dict['total_loss'], global_step)

        return train_op, loss_dict

    def test(self, test_data):
        with tf.name_scope('test'):
            logits = self.forward_pass(test_data['images'], trainable=False)
            tf.summary.image('test_img', test_data['images'])

            top_k_op = tf.nn.in_top_k(logits, test_data['labels'], 1)

        return top_k_op
