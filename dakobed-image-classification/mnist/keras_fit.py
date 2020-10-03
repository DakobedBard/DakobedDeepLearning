import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from tqdm import tqdm
from utils import plot_image_prediction, plot_value_prediction, LossHistory, PeriodicPlotter

from cnn import build_cnn_model

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)

BATCH_SIZE = 64
EPOCHS = 5

model = build_cnn_model()

'''TODO: Define the compile operation with your optimizer and learning rate of choice'''
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

'''TODO: Use the evaluate method to test the model!'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
# test_loss, test_acc = # TODO

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

