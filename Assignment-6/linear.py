import tensorflow as tf

a=tf.Variable([0.2],dtype=tf.float32)
b=tf.Variable([-0.2],dtype=tf.float32)
x=tf.placeholder(tf.float32)

y=tf.placeholder(tf.float32)

linear_mod=a*x+b

loss=tf.reduce_sum(tf.square(linear_mod-y))

optimizer=tf.train.GradientDescentOptimizer(0.01)

t=optimizer.minimize(loss)

x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]

init=tf.global_variables_initializer()

with tf.Session() as s:
    s.run(init)

    for i in range(1000):
     s.run(t,{x:x_train,y:y_train})

    cur_a,cur_b,cur_loss = s.run([a,b,loss],{x:x_train,y:y_train})
    print("a: %s b: %s loss: %s"%(cur_a,cur_b,cur_loss))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
