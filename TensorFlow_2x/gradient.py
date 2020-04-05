import tensorflow as tf

x = tf.constant(1.0)
a = tf.constant(2.0)
b = tf.constant(3.0)
c = tf.constant(4.0)

with tf.GradientTape() as tape:
    tape.watch([a,b,c,x])
    y = a**2 * x + b * x + c

[dy_da, dy_db, dy_dc, dy_dx] = tape.gradient(y,[a,b,c,x])
print(dy_da, dy_db,dy_dc,dy_dx)

# results:
# tf.Tensor(4.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32) tf.Tensor(7.0, shape=(), dtype=float32)