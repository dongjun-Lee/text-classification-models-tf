import tensorflow as tf


class VDCNN(object):
    def __init__(self, alphabet_size, document_max_len, num_class):
        self.embedding_size = 16
        self.filter_sizes = [3, 3, 3, 3, 3]
        self.num_filters = [64, 64, 128, 256, 512]
        self.num_blocks = [2, 2, 2, 2]
        self.learning_rate = 1e-3
        self.cnn_initializer = tf.keras.initializers.he_normal()
        self.fc_initializer = tf.truncated_normal_initializer(stddev=0.05)

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)

        # ============= Embedding Layer =============
        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([alphabet_size, self.embedding_size], -1.0, 1.0)
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)
            self.x_expanded = tf.expand_dims(x_emb, -1)

        # ============= First Convolution Layer =============
        with tf.variable_scope("conv-0"):
            conv0 = tf.layers.conv2d(
                self.x_expanded,
                filters=self.num_filters[0],
                kernel_size=[self.filter_sizes[0], self.embedding_size],
                kernel_initializer=self.cnn_initializer,
                activation=tf.nn.relu)
            conv0 = tf.transpose(conv0, [0, 1, 3, 2])

        # ============= Convolution Blocks =============
        with tf.name_scope("conv-block-1"):
            conv1 = self.conv_block(conv0, 1)

        with tf.name_scope("conv-block-2"):
            conv2 = self.conv_block(conv1, 2)

        with tf.name_scope("conv-block-3"):
            conv3 = self.conv_block(conv2, 3)

        with tf.name_scope("conv-block-4"):
            conv4 = self.conv_block(conv3, 4, max_pool=False)

        # ============= k-max Pooling =============
        with tf.name_scope("k-max-pooling"):
            h = tf.transpose(tf.squeeze(conv4, -1), [0, 2, 1])
            top_k = tf.nn.top_k(h, k=8, sorted=False).values
            h_flat = tf.reshape(top_k, [-1, 512 * 8])

        # ============= Fully Connected Layers =============
        with tf.name_scope("fc-1"):
            fc1_out = tf.layers.dense(h_flat, 2048, activation=tf.nn.relu, kernel_initializer=self.fc_initializer)

        with tf.name_scope("fc-2"):
            fc2_out = tf.layers.dense(fc1_out, 2048, activation=tf.nn.relu, kernel_initializer=self.fc_initializer)

        with tf.name_scope("fc-3"):
            self.logits = tf.layers.dense(fc2_out, num_class, activation=None, kernel_initializer=self.fc_initializer)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        # ============= Loss and Accuracy =============
        with tf.name_scope("loss"):
            y_one_hot = tf.one_hot(self.y, num_class)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_one_hot))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def conv_block(self, input, i, max_pool=True):
        with tf.variable_scope("conv-block-%s" % i):
            # Two "conv-batch_norm-relu" layers.
            for j in range(2):
                with tf.variable_scope("conv-%s" % j):
                    # convolution
                    conv = tf.layers.conv2d(
                        input,
                        filters=self.num_filters[i],
                        kernel_size=[self.filter_sizes[i], self.num_filters[i-1]],
                        kernel_initializer=self.cnn_initializer,
                        activation=None)
                    # batch normalization
                    conv = tf.layers.batch_normalization(conv, training=self.is_training)
                    # relu
                    conv = tf.nn.relu(conv)
                    conv = tf.transpose(conv, [0, 1, 3, 2])

            if max_pool:
                # Max pooling
                pool = tf.layers.max_pooling2d(
                    conv,
                    pool_size=(3, 1),
                    strides=(2, 1),
                    padding="SAME")
                return pool
            else:
                return conv
