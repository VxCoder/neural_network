# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 50  # 训练数据个数. 数据越大-> 梯度下降   数据越小-> 随机梯度下降

LEARNING_RATE_BASE = 0.01  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZTION_RATE = 0.0001  # 正则化系数
TRAINING_STEPS = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def train(mnist):

    # x = tf.placeholder(tf.float32, shape=[None, mnist_inference.INPUT_NODE], name='x-input')  # 输入

    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')

    y_ = tf.placeholder(tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name='y_input')  # 标签

    # 正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZTION_RATE)

    # 前向传播
    # y = mnist_inference.inference(x, regularizer=regularizer)  # 全连接神经网络
    y = mnist_inference.cnn_inference(x, train=True, regularizer=regularizer)  # cnn神经网络

    global_step = tf.Variable(0, trainable=False)

    # 滑动平滑窗口
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 带正常化的损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 反向传播 + 滑动平均值
    train_op = tf.group(train_step, variable_averages_op)

    # 持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs = np.reshape(xs, (BATCH_SIZE,
                                 mnist_inference.IMAGE_SIZE,
                                 mnist_inference.IMAGE_SIZE,
                                 mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})

            if i % 10 != 0:
                continue

            # 显示模式损失函数大小,并保存模型
            print(f"After {step} trainning step(s), loss on training batch is {loss_value}")
            saver.save(
                sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('.', one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
