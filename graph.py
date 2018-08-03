import tensorflow as tf


def main():
    graph1 = tf.Graph()
    with graph1.as_default():
        tf.get_variable(
            "v", initializer=tf.zeros_initializer()(shape=[1])
        )

    graph2 = tf.Graph()
    with graph2.as_default():
        tf.get_variable(
            "v", initializer=tf.ones_initializer()(shape=[1])
        )

    with tf.Session(graph=graph1) as sess:
        tf.global_variables_initializer().run()
        with tf.variable_scope("", reuse=True):
            print(sess.run(tf.get_variable("v")))

    with tf.Session(graph=graph2) as sess:
        tf.global_variables_initializer().run()
        with tf.variable_scope("", reuse=True):
            print(sess.run(tf.get_variable("v")))


if __name__ == "__main__":
    main()
