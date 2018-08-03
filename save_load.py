import tensorflow as tf


def simple_save():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2
    saver = tf.train.Saver()
    saver.export_meta_graph("./json/model.ckpt.json", as_text=True)


def save():
    v = tf.Variable(0, dtype=tf.float32, name="v")

    ema = tf.train.ExponentialMovingAverage(0.99)
    maintain_averages_op = ema.apply(tf.all_variables())

    for variable in tf.all_variables():
        print(variable.name)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(v, 10))
        sess.run(maintain_averages_op)
        saver.save(sess, "./model/model.ckpt")
        print(sess.run([v, ema.average(v)]))


def load():
    v = tf.Variable(0, dtype=tf.float32, name="v")
    ema = tf.train.ExponentialMovingAverage(0.99)
    saver = tf.train.Saver(ema.variables_to_restore())

    with tf.Session() as sess:
        saver.restore(sess, "./model/model.ckpt")
        print(sess.run(v))


from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def more_save():

    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph_def, ['add'])
        with tf.gfile.GFile('./model/combind_model.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())


def more_load():
    with tf.Session() as sess:
        model_filename = "./model/combind_model.pb"
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        result = tf.import_graph_def(graph_def, return_elements=["add:0"])
        print(sess.run)


def main():
    simple_save()
    # save()
    # load()


if __name__ == "__main__":
    main()
