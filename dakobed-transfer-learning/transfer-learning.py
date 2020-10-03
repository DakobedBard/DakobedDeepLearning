import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

split = (80, 10, 10)
(cat_train, cat_valid, cat_test), info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[10%:]', 'train[10%:]'], with_info=True, as_supervised=True)

import matplotlib.pylab as plt
for image, label in cat_train.take(2):
  plt.figure()
  plt.imshow(image)

IMAGE_SIZE = 100
def pre_process_image(image, label):
  image = tf.cast(image, tf.float32)
  image = image / 255.0
  image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
  return image, label


TRAIN_BATCH_SIZE = 64
cat_train = cat_train.map(pre_process_image).shuffle(1000).repeat().batch(TRAIN_BATCH_SIZE)
cat_valid = cat_valid.map(pre_process_image).repeat().batch(1000)


IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
res_net = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)


global_average_layer = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(1, activation='sigmoid')
tl_model = tf.keras.Sequential([
  res_net,
  global_average_layer,
  output_layer
])

tl_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

tl_model.fit(cat_train, steps_per_epoch = 23262//TRAIN_BATCH_SIZE, epochs=7,  validation_data=cat_valid, validation_steps=10)