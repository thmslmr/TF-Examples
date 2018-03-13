# Simple regression example using Eager execution

# Imports
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt
import utils

# Define paramaters
LEARNING_RATE = 0.01
N_EPOCHS = 100
LOSS_TYPE = 'HUBER'

# Call `tfe.enable_eager_execution` to use eager execution
tfe.enable_eager_execution()

# Read the data into a dataset.
DATA_FILE = '../data/birth_life.txt'
data, n_samples = utils.read_data_file(DATA_FILE)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))

# Create variables
w = tfe.Variable(0.0)
b = tfe.Variable(0.0)

# Define the linear predictor.
def prediction(x):
  return x * w + b

# Define squared loss function
def squared_loss(y, y_predicted):
  return (y - y_predicted) ** 2

# Define huber loss function
def huber_loss(y, y_predicted, m=1.0):
  t = y - y_predicted
  return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

# Define train function
def train(loss_fn):

    # Use gradient descent to minimize loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)

    # Define the function through which to differentiate.
    def loss_for_example(x, y):
        return loss_fn(y, prediction(x))

    # `grad_fn(x_i, y_i)` returns (1) the value of `loss_for_example`
    # evaluated at `x_i`, `y_i` and (2) the gradients of any variables used in
    # calculating it.
    grad_fn = tfe.implicit_value_and_gradients(loss_for_example)

    start = time.time()
    for epoch in range(100):
        total_loss = 0.0

        for x_i, y_i in tfe.Iterator(dataset):
          loss, gradients = grad_fn(x_i, y_i)
          optimizer.apply_gradients(gradients)
          total_loss += loss

        if epoch % 10 == 0:
          print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))

    print('Took: {0} seconds'.format(time.time() - start))

train(huber_loss)

plt.plot(data[:,0], data[:,1], 'bo')
plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r',
    label="huber regression")
plt.legend()
plt.show()
