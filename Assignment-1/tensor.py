import tensorflow as tf
from google.datalab.ml import TensorBoard as tb

a = tf.constant(6.5,name='con_a')
b = tf.constant(5.5,name='con_b')
c = tf.constant(3.6,name='con_c')
d = tf.constant(5.6,name='con_d')

add = tf.add(a,b,name='add_ab')
sub = tf.subtract(b,c,name='sub_bc')
sq = tf.square(d,name='sq_d')

final_sum = tf.add_n([add,sub,sq],name='add_another')

another = tf.add_n([a,b,c,d,sq],name='add_all')

with tf.Session() as s:
    print("a+b:", s.run(add))
    print("b-c:", s.run(sub))
    print("Square of D:", s.run(sq))
    print("All sum :", s.run(final_sum))
    print("Sum of all:", s.run(another))

    writer =tf.summary.FileWriter('./tensor',s.graph)
    writer.close();
tensorb = tb.start('./tensor')
tb.stop(tensorb)
