# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x-input')  # 输入
        y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y_input')  # 标签

        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

            if ckpt and ckpt.all_model_checkpoint_paths:
                for all_model_checkpoint_path in ckpt.all_model_checkpoint_paths:
                    saver.restore(sess, all_model_checkpoint_path)
                    global_step = all_model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print(f"After {global_step} traning step(s), validation accuracy = {accuracy_score}")


def main():
    mnist = input_data.read_data_sets('.', one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    main()
