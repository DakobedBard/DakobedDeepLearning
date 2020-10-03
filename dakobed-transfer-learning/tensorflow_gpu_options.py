import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time



NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))
tf.config.experimental.list_physical_devices('GPU')

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

