# Simple Linear Regression example

# Imports
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils

# Define paramaters
LEARNING_RATE = 0.001
N_EPOCHS = 100
LOSS_TYPE = 'HUBER'

# Read in the data
DATA_FILE = '../data/birth_life.txt'
data, n_samples = utils.read_data_file(DATA_FILE)

# Create Dataset and iterator
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

# Create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Build model to predict Y
Y_predicted = w * X + b

# Use either square (type = 'SQUARE') or Huber (type = 'HUBER') loss function
loss = tf.square(Y - Y_predicted, name='loss') if LOSS_TYPE == 'SQUARE'
    else utils.huber_loss(Y, Y_predicted)

# Use gradient descent to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

start = time.time()
with tf.Session() as sess:

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Create writer for TensorBoard
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # Train the model for 100 epochs
    for i in range(N_EPOCHS):
        # Initialize the iterator
        sess.run(iterator.initializer)
        total_loss = 0
        try:
            while True:
                _ , l = sess.run([optimizer, loss])
                total_loss += l
        except tf.errors.OutOfRangeError:
            pass

        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    # Close the writer when you're done using it
    writer.close()

    # Output the values of w and b
    w_out, b_out = sess.run([w, b])
    print('Linear prediction equ : {0} * X + {1}'.format(w_out, b_out))
print('Took: {0} seconds'.format(time.time() - start))

# Plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data with squared error')
plt.legend()
plt.show()
