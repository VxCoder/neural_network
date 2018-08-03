import tensorflow as tf
from numpy.random import RandomState

BATCH_SIZE = 8


def main():

    # 两层权重参数
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # 输入层数据
    x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="x-input")

    # 标签数据
    y_ = tf.placeholder(tf.float32,  shape=(None, 1), name='y-input')

    # 前向传播算法
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # 损失函数
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    # 反向传播算法
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 生成模拟数据集
    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2)
    Y = [[int(x1 + x2) < 1] for (x1, x2) in X]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(w1))
        print(sess.run(w2))

        STEPS = 5000
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % dataset_size
            end = min(start + BATCH_SIZE, dataset_size)

            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

            if i % 1000 == 0:
                total_cross_entryopy = sess.run(
                    cross_entropy, feed_dict={x: X, y_: Y}
                )
                print(f"After {i:0<4} tranining setp(s), cross entropy on all data is {total_cross_entryopy}")

        print(sess.run(w1))
        print(sess.run(w2))


if __name__ == "__main__":
    main()
