import numpy as np
import tensorflow as tf
import math
W = tf.Variable(tf.zeros([4, 4]))
b = tf.Variable(tf.zeros([4])
x = tf.placeholder("float", None)

y = tf.placeholder("float", 4)

model=tf.nn.softmax(tf.matmul(x, W) + b)
#model=a*tf.pow(x1,n1)+b*tf.pow(x2,n2)+c*tf.pow(x3,n3)+d*tf.pow(x4,n4)

#loss = tf.reduce_sum(tf.square(model - y))  #algebraic generalised regression
loss = -tf.reduce_sum(y*tf.log(model))


optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
x_train = [[1, 2, 3 4],
              [4, 5, 6 6],
              [3, 4, 5, 7],
              [4, 5, 6, 8]]         ]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(10000):
  sess.run(train, feed_dict{x:x_train, y:y_train} )

# evaluate training accuracy
curr_w,curr_b, curr_loss  = sess.run([W,b,loss], {x:x_train,y:y_train})
print("W: %s,b: %s, loss: %s"%(curr_w,curr_b,curr_loss))
