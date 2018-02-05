import tensorflow as tf
from google.datalab.ml import TensorBoard as tb

x = tf.constant([12,24,36],name='x')
y = tf.constant([100,200,300],name='y')

sum_x = tf.reduce_sum(x,name='sum')
prod = tf.reduce_prod(y,name='prod')

final_div = tf.div(sum_x,prod,name='f_div')

final_mean = tf.reduce_mean([sum_x,prod],name='f_mean')

with tf.Session() as s:
    
    print "x=",s.run(x)
    print "y=",s.run(y)
    print "sum=",s.run(sum_x)
    print "product=",s.run(prod)
    print "final_division=",s.run(final_div)
    print "final_mean=",s.run(final_mean)

    writer=tf.summary.FileWriter('./rank',s.graph)
    writer.close()

tb.start('./rank')

