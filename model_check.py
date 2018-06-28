import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import math
a = tf.Variable([.3], tf.float32)
b = tf.Variable([.5], tf.float32)
c = tf.Variable([.6], tf.float32)
d = tf.Variable([.2], tf.float32)
n1 = tf.Variable([.3], tf.float32)
n2 = tf.Variable([.5], tf.float32)
n3 = tf.Variable([.6], tf.float32)
n4 = tf.Variable([.2], tf.float32)
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
x4 = tf.placeholder(tf.float32)
f =  tf.Variable([.3], tf.float32)
g =  tf.Variable([.3], tf.float32)
h =  tf.Variable([.3], tf.float32)
k =  tf.Variable([.3], tf.float32)

#model=a*x1+b*x2+c*x3+d*x4

model=100000/(100+900*tf.exp(-a*x1))
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(model - y))  #algebraic generalised regression
#loss = -tf.reduce_sum(y*tf.log(model))


optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
x1_train = np.loadtxt('pop.txt', usecols=(0,))
#x2_train = np.loadtxt('input_norm.txt', usecols=(1,))
#x3_train = np.loadtxt('input_norm.txt', usecols=(2,))
#x4_train = np.loadtxt('input_norm.txt', usecols=(3,))
y_train  =  np.loadtxt('pop.txt',usecols=(1,))
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
#  sess.run(train, {x1:x1_train,x2:x2_train,x3:x3_train,x4:x4_train, y:y_train})
  sess.run(train, {x1:x1_train, y:y_train})
#print(x2_train)
# evaluate training accuracy
#curr_a, curr_b, curr_c, curr_d, curr_n1, curr_n2, curr_n3, curr_n4, curr_loss  = sess.run([a, b, c, d,n1,n2,n3,n4,loss], {x1:x1_train,x2:x2_train,x3:x3_train,x4:x4_train, y:y_train})
curr_a, curr_loss  = sess.run([a,loss], {x1:x1_train, y:y_train})
#print("a: %s b: %s c: %s d: %s  n1: %s n2: %s n3: %s n4: %s loss: %s"%(curr_a, curr_b, curr_c,curr_d,curr_n1,curr_n2,curr_n3,curr_n4,curr_loss))
print("a: %s  loss: %s"%(curr_a,curr_loss))
