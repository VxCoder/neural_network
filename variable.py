import tensorflow as tf

# zero = tf.zeros([2, 3])
# one = tf.ones([2, 3])
# fill = tf.fill([2, 3], 9)
# constant = tf.constant([1, 2, 3])
# normal = tf.Variable(tf.random_normal([2, 2], mean=0, stddev=2, dtype=tf.float16))
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(zero.eval())
#     print(one.eval())
#     print(fill.eval())
#     print(constant.eval())
#     print(normal.eval())

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v1.name)
