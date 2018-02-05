import tensorflow as tf
from google.datalab.ml import TensorBoard as tb

x = tf.placeholder(tf.int32,shape=[3],name='x')
y = tf.placeholder(tf.int32,shape=[3],name='y')

sum_x = tf.reduce_sum(x,name='sum_x')
prod = tf.reduce_prod(y,name='prod')

div = tf.div(sum_x , prod,name='divide')

with tf.Session() as s:


    print'sum of x',s.run(sum_x,feed_dict={x:[100,200,300]})
    print'Prod of y',s.run(prod,feed_dict={y:[11,22,33]})
    print'Division :',s.run(div,feed_dict={x:[100,200,300],y:[11,22,33]})

    writer = tf.summary.FileWriter('./Placeholder',s.graph)
    writer.close()

tb.start('./Placeholder')


