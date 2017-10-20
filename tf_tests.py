import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Just adding two nodes
sess = tf.Session()
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # substitutes tf.add(a, b)
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# Calculating linear loss
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(
