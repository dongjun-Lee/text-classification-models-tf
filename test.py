import tensorflow as tf
import argparse
from data_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="word_cnn",
                    help="word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn")
args = parser.parse_args()

BATCH_SIZE = 128
WORD_MAX_LEN = 100
CHAR_MAX_LEN = 1014

if args.model == "char_cnn":
    test_x, test_y, alphabet_size = build_char_dataset("test", "char_cnn", CHAR_MAX_LEN)
elif args.model == "vd_cnn":
    test_x, test_y, alphabet_size = build_char_dataset("test", "vdcnn", CHAR_MAX_LEN)
else:
    word_dict = build_word_dict()
    test_x, test_y = build_word_dataset("test", word_dict, WORD_MAX_LEN)

checkpoint_file = tf.train.latest_checkpoint(args.model)
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        batches = batch_iter(test_x, test_y, BATCH_SIZE, 1)
        sum_accuracy, cnt = 0, 0
        for batch_x, batch_y in batches:
            feed_dict = {
                x: batch_x,
                y: batch_y,
                is_training: False
            }

            accuracy_out = sess.run(accuracy, feed_dict=feed_dict)
            sum_accuracy += accuracy_out
            cnt += 1

        print("Test Accuracy : {0}".format(sum_accuracy / cnt))
