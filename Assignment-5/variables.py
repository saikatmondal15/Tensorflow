import tensorflow as tf
from google.datalab.ml import TensorBoard as tb

a=tf.Variable([2.5,3.5],tf.float32,name='a')
b=tf.Variable([2.1,5.0],tf.float32,name='b')
x=tf.placeholder(tf.float32,name='x')
y=a+b*x

number = tf.Variable(10)
mult = tf.Variable(1)

init = tf.global_variables_initializer()
result = number.assign(tf.multiply(number,mult))
with tf.Session() as s:
    s.run(init)
    print 'Value of y:',s.run(y,feed_dict={x:[10,100]})
    for i in range(5):
        print 'Result =',s.run(result)
        print 'Increment multiplyer=',s.run(mult.assign_add(1))





