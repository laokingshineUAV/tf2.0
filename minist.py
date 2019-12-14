import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, metrics

(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255.0 - 1
y = tf.convert_to_tensor(y, dtype=tf.int32)
#y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)


train_data = tf.data.Dataset.from_tensor_slices((x, y))
db = train_data.batch(32).repeat(10)

model = keras.Sequential(
    [
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ]
)
model.build(input_shape=(None, 28*28))
model.summary()
optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()

for step, (x, y) in enumerate(db):

    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28*28))
        out = model(x)

        y_one_hot = tf.one_hot(y, depth=10)
        loss = tf.square(out - y_one_hot)
        loss = tf.reduce_sum(loss) / 32

    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step % 200==0:

        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()