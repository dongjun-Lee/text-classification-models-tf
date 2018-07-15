import tensorflow as tf
from tensorflow.contrib import rnn


class RCNN(object):
    def __init__(self, vocabulary_size, document_max_len, num_class):
        self.embedding_size = 256
        self.rnn_num_hidden = 256
        self.fc_num_hidden = 256
        self.learning_rate = 1e-3

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)

        with tf.name_scope("birnn"):
            fw_cell = rnn.BasicLSTMCell(self.rnn_num_hidden)
            bw_cell = rnn.BasicLSTMCell(self.rnn_num_hidden)
            fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
            bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)

            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.x_emb,
                                                             sequence_length=self.x_len, dtype=tf.float32)
            self.fw_output, self.bw_output = rnn_outputs

        with tf.name_scope("word-representation"):
            x = tf.concat([self.fw_output, self.x_emb, self.bw_output], axis=2)
            self.y2 = tf.layers.dense(x, self.fc_num_hidden, activation=tf.nn.tanh)

        with tf.name_scope("text-representation"):
            self.y3 = tf.reduce_max(self.y2, axis=1)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(self.y3, num_class)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
