# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import mnist_inference
import mnist_train


def load(input_data):
    input_data = np.reshape(input_data, (1, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[1, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')  # 输入

        y = mnist_inference.cnn_inference(x, False, None)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        result = tf.argmax(y, 1)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

            if ckpt and ckpt.all_model_checkpoint_paths:
                checkpoint_path = ckpt.all_model_checkpoint_paths[-1]
                saver.restore(sess, checkpoint_path)
                result = sess.run(result, feed_dict={x: input_data})

                return result
