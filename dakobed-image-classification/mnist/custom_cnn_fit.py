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


cnn_model = build_cnn_model()

batch_size = 12
loss_history = LossHistory(smoothing_factor=0.95) # to record the evolution of the loss
plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss', scale='semilogy')
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2) # define our optimizer

if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for idx in tqdm(range(0, train_images.shape[0], batch_size)):
    # First grab a batch of training data and convert the input images to tensors
    (images, labels) = (train_images[idx:idx+batch_size], train_labels[idx:idx+batch_size])
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    # GradientTape to record differentiation operations
    with tf.GradientTape() as tape:
        logits = cnn_model(images)
        # logits = # TODO

        loss_value = tf.keras.backend.sparse_categorical_crossentropy(labels, logits)
        # loss_value = tf.keras.backend.sparse_categorical_crossentropy() # TODO

    loss_history.append(loss_value.numpy().mean()) # append the loss to the loss_history record
    plotter.plot(loss_history.get())

    # Backpropagation
    '''TODO: Use the tape to compute the gradient against all parameters in the CNN model.
        Use cnn_model.trainable_variables to access these parameters.'''
    grads = tape.gradient(loss_value, cnn_model.trainable_variables)
    # grads = # TODO
    optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))