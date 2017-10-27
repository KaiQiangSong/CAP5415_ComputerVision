import time


import tensorflow as tf

########### Convolutional neural network class ############
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        with tf.name_scope('reshape'):
            X = tf.reshape(X, [-1, 784])
            
        with tf.name_scope('fc'):
            W_fc = weight_variable([784, hidden_size])
            b_fc = bias_variable([hidden_size])
        
        return tf.nn.sigmoid(tf.matmul(X, W_fc) + b_fc)

    # Use two convolutional layers.
    def model_2(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 40])
            b_conv1 = bias_variable([40])
            h_conv1 = tf.nn.sigmoid(conv2d(X, W_conv1) + b_conv1)
        
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)
            
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 40, 40])
            b_conv2 = bias_variable([40])
            h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)  
                  
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)
            
        with tf.name_scope('fc'):
            W_fc = weight_variable([7 * 7 * 40, hidden_size])
            b_fc = bias_variable([hidden_size])
            
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
            h_fc = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc) + b_fc)
        
        return h_fc

    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 40])
            b_conv1 = bias_variable([40])
            h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
        
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)
            
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 40, 40])
            b_conv2 = bias_variable([40])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
                  
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)
            
        with tf.name_scope('fc'):
            W_fc = weight_variable([7 * 7 * 40, hidden_size])
            b_fc = bias_variable([hidden_size])
            
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
            h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)
        return h_fc

    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.     
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 40])
            b_conv1 = bias_variable([40])
            h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
        
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)
            
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 40, 40])
            b_conv2 = bias_variable([40])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
                  
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)
            
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 40, hidden_size])
            b_fc1 = bias_variable([hidden_size])
            
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            
            L2_fc1 = tf.nn.l2_loss(W_fc1)
        
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([hidden_size, hidden_size])
            b_fc2 = bias_variable([hidden_size])
            
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
            L2_fc2 = tf.nn.l2_loss(W_fc2)
            
        return h_fc2, decay * (L2_fc1 + L2_fc2)

    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 40])
            b_conv1 = bias_variable([40])
            h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
        
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)
            
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 40, 40])
            b_conv2 = bias_variable([40])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
                  
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)
            
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 40, hidden_size])
            b_fc1 = bias_variable([hidden_size])
            
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            
            
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)
        
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([hidden_size, hidden_size])
            b_fc2 = bias_variable([hidden_size])
            
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        return h_fc2

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        print batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)

        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.int32, [None])
            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X, hidden_size)



            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)


            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features, L2 = self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train)

            # ======================================================================
            # Define softmax layer, use the features.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to logits after code implementation.
            W = tf.Variable(tf.zeros([hidden_size, class_num]))
            b = tf.Variable(tf.zeros([class_num]))
            logits = tf.matmul(features, W) + b

            # ======================================================================
            # Define loss function, use the logits.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to loss after code implementation.
            y_ = tf.one_hot(Y, class_num)
            if self.mode != 4:
                L2 = 0
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_)) + L2

            # ======================================================================
            # Define training op, use the loss.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to train_op after code implementation.
            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # ======================================================================
            # Define accuracy op.
            # ----------------- YOUR CODE HERE ----------------------
            #
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = True
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    start_time = time.time()
                    s = 0
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]
                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))
                    
                    total_correct = 0
                    s = 0
                    while s < test_size:
                        e = min(s + batch_size, test_size)
                        batch_x = testX[s: e]
                        batch_y = testY[s: e]
                        total_correct += sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, is_train: False})
                        s = e
                    print ('accuracy of the trained model %f' % (total_correct / testX.shape[0]))
                    print ()
                total_correct = 0
                s = 0
                while s < test_size:
                    e = min(s + batch_size, test_size)
                    batch_x = testX[s: e]
                    batch_y = testY[s: e]
                    total_correct += sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, is_train: False})
                    s = e
                return total_correct / testX.shape[0]





