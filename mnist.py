# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.distribute.python import values


INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500  # 有500节点的隐藏层
BATCH_SIZE = 100  # 训练数据个数. 数据越大-> 梯度下降   数据越小-> 随机梯度下降

LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZTION_RATE = 0.0001  # 正则化系数
TRAINING_STEPS = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def avg_values(avg_class, *values):
    if not avg_class:
        return values
    else:
        return [avg_class.average(value) for value in values]


def inference(input_tensor, reuse=False, avg_class=None, regularizer=None):

    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights", shape=[INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        if regularizer:
            tf.add_to_collection('losses', regularizer(weights))

        weights, biases = avg_values(avg_class, weights, biases)

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", shape=[LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))

        if regularizer:
            tf.add_to_collection('losses', regularizer(weights))

        weights, biases = avg_values(avg_class, weights, biases)
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name='x-input')  # 输入
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name='y_input')  # 标签

    # 正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZTION_RATE)

    # 前向传播
    y = inference(x, regularizer=regularizer)

    # 带滑动平滑的前向传播
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, reuse=True, avg_class=variable_averages)

    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'))

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

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print(f"After {i} trainning step(s), validation accuracy using average model is {validate_acc}")

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(f"After {TRAINING_STEPS} training step(s), test accuracy using average model is {test_acc}")


def main(argv=None):
    mnist = input_data.read_data_sets('.', one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
