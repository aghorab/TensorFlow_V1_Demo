import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant(-2.0, name="x", dtype=tf.float32)
a = tf.constant(5.0, name="a", dtype=tf.float32)
b = tf.constant(13.0, name="b", dtype=tf.float32)

y = tf.Variable(tf.add(tf.multiply(a, x), b))
h = tf.nn.relu(y)

init = tf.global_variables_initializer()

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", session.graph)

    session.run(init)
    print(session.run(h))
