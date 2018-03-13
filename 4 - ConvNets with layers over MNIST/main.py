import os
import time
import tensorflow as tf
import utils

# Define ConvNet class for reusable code
class ConvNet(object):

    # Init class
    def __init__(self):
        self.data_path = '../data/mnist'
        self.learning_rate = 0.001
        self.batch_size = 128
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                trainable=False, name='global_step')
        self.n_classes = 10
        self.skip_step = 20
        self.n_test = 10000
        self.training = False

    # Get MNIST data
    def get_data(self):
        with tf.name_scope('data'):
            # Fetch data
            train_data, test_data = utils.get_mnist_dataset(
                self.data_path, self.batch_size)

            # Create Iterator to get samples from the two dataset
            iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)
            img, self.label = iterator.get_next()

            # Reshape the image to make it work with tf.nn.conv2d
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])

            # Initializer for train and test data
            self.train_init = iterator.make_initializer(train_data)
            self.test_init = iterator.make_initializer(test_data)

    # Define model structure
    def inference(self):

        # Use tf.layers functions for conv maxpool and fc
        conv1 = tf.layers.conv2d(inputs=self.img,
                                  filters=32,
                                  kernel_size=[5, 5],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='conv1')

        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name='pool1')

        conv2 = tf.layers.conv2d(inputs=pool1,
                                  filters=64,
                                  kernel_size=[5, 5],
                                  padding='SAME',
                                  activation=tf.nn.relu,
                                  name='conv2')

        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name='pool2')

        # Get max pooling 2 layer dimension and reshape for FC layer
        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])

        fc = tf.layers.dense(pool2, 1024, activation=tf.nn.relu, name='fc')

        # Dropout only during training
        dropout = tf.layers.dropout(fc,
                                    self.keep_prob,
                                    training=self.training,
                                    name='dropout')
        # Final value of the model
        self.logits = tf.layers.dense(dropout, self.n_classes, name='logits')

    # Define loss function
    def loss(self):

        # Use softmax cross entropy with logits as the loss function
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    # Define training op
    def optimize(self):

        # Use Adam Gradient Descent to minimize cost
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                global_step=self.gstep)

    # Create summary for TensorBoard
    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    # Get number of right predictions
    def eval(self):
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    # Build our learning model structure
    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.eval()
        self.summary()

    # Define training for one epoch
    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)

                # Print loss every 20 step
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1

                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, 'checkpoints/convnet_layers/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds/self.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    # Main train alternates training one epoch and evaluating
    def train(self, n_epochs):
        utils.safe_mkdir('checkpoints')
        utils.safe_mkdir('checkpoints/convnet_layers')

        writer = tf.summary.FileWriter('./graphs/convnet_layers', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/checkpoint'))

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()


if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=15)
