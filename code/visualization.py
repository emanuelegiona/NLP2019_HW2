from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from gensim.models import keyedvectors


def make_tensorboard_model(embeddings_path, output_path):
    """
    Build an embedding-only Tensorflow model in order to use Tensorboard Projector for embeddings.
    :param embeddings_path: Path to the embeddings.vec file
    :param output_path: Path to the folder where to store both the Tensorflow model and TSV file for labels
    :return: None
    """

    print("Reading embeddings...")
    emb_matrix = keyedvectors.KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
    vocab = emb_matrix.vocab

    nparr = np.ndarray([len(vocab), emb_matrix.vector_size])
    for i, word in enumerate(vocab.keys()):
        nparr[i] = emb_matrix.get_vector(word)

    embeddings = tf.get_variable(name="embeddings",
                                 shape=[nparr.shape[0], nparr.shape[1]],
                                 initializer=tf.constant_initializer(nparr),
                                 trainable=False)

    print("Exporting Tensorflow model...")
    # save model for TensorBoard use
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, output_path + "/embeddings.ckpt")

    print("Exporting labels as a .tsv file...")
    # export TSV file for easier reading on TensorBoard
    with open(output_path + "/labels.tsv", mode="w") as tsv:
        for word in vocab.keys():
            tsv.write("%s\n" % word)
            tsv.flush()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("embeddings_path", help="Path to the embeddings.vec file")
    parser.add_argument("output_path", help="Path to the output folder")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_tensorboard_model(embeddings_path=args.embeddings_path,
                           output_path=args.output_path)
