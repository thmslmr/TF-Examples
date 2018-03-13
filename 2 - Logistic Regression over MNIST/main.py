# MNIST Logistic Regression

# Imports
import numpy as np
import tensorflow as tf
import time
import utils

# Define paramaters
LEARNING_RATE = 0.01
BATCH_SIZE = 128
N_EPOCHS = 30
N_TRAIN = 6000
N_TEST = 10000

# Read in data
MNIST_FOLDER = '../data/mnist'
utils.download_mnist(MNIST_FOLDER)
train, val, test = utils.read_mnist(MNIST_FOLDER, flatten=True)

# Create train and test dataset
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
test_data = tf.data.Dataset.from_tensor_slices(test)

# Process the data in batches
train_data = train_data.batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

# Create Iterator to get samples from the two dataset
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

# Initializer for train and test data
train_init = iterator.make_initializer(train_data)
test_init = iterator.make_initializer(test_data)

# Create weights and bias
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer(0, 0.01))
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

# Build the model
logits = tf.matmul(img, w) + b

# Define loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')

# Define training op
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# Calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

start_time = time.time()
with tf.Session() as sess:

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Create writer for TensorBoard
    writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

    # Train the model n_epochs times
    for i in range(N_EPOCHS):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # Test the model
    sess.run(test_init)
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/N_TEST))
    writer.close()
