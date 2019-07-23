import tensorflow as tf

from model import Word2Vec, SynsetAwareWord2Vec
import utils as u

# --- global variables ---
EMBEDDING_SIZE = 64
LEARNING_RATE = 0.15
WINDOW_SIZE = 3
NEG_SAMPLES = 16
MODEL_PATH_W2V = "../resources/models/basic_w2v/model.ckpt"
MODEL_PATH_SYN_W2V = "../resources/models/synaware_w2v/model.ckpt"
# --- --- ---


def evaluate(target_words, top_k=10, synaware_w2v=True):
    """
    Provides a qualitative measure for the embeddings the model has learned by printing the most similar words to the
    ones provided as test words.
    :param target_words: Test words to discover the closest words to them, as List
    :param top_k: Number of closest words
    :param synaware_w2v: True: use a SynsetAwareWord2Vec model (default); False: use a basic Word2Vec model
    :return: None
    """

    print("Loading vocabularies...")
    word_to_ix, ix_to_word, subsampled_words = u.get_vocab(vocab_path="../resources/vocab.txt",
                                                           antivocab_path="../resources/antivocab.txt")

    print("Creating model...")
    if not synaware_w2v:
        model = Word2Vec(subsampled_words=subsampled_words,
                         vocabulary_size=len(word_to_ix),
                         embedding_size=EMBEDDING_SIZE,
                         learning_rate=LEARNING_RATE,
                         window_size=WINDOW_SIZE,
                         neg_samples=NEG_SAMPLES)
    else:
        model = SynsetAwareWord2Vec(subsampled_words=subsampled_words,
                                    vocabulary_size=len(word_to_ix),
                                    embedding_size=EMBEDDING_SIZE,
                                    learning_rate=LEARNING_RATE,
                                    window_size=WINDOW_SIZE,
                                    neg_samples=NEG_SAMPLES)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Loading model...")
        saver.restore(sess, MODEL_PATH_SYN_W2V if synaware_w2v else MODEL_PATH_W2V)

        target_words = [word_to_ix[w] for w in target_words if w in word_to_ix]
        sim_val = sess.run(model.similarity,
                           feed_dict={model.data["sim_test"]: target_words})

        for i in range(len(target_words)):
            print("Closest %d words to %s:" % (top_k, ix_to_word[target_words[i]]))
            closest_words = (-sim_val[i, :]).argsort()[1:top_k + 1]
            for j in range(top_k):
                word = ix_to_word[closest_words[j]]
                print("\t%d. %s" % (j+1, word))


if __name__ == "__main__":
    tf.reset_default_graph()
    evaluate(["Europe_bn:00031896n", "France_bn:00036202n", "Council"], synaware_w2v=False)
    print("\n---\t---\t---\n")
    tf.reset_default_graph()
    evaluate(["Europe_bn:00031896n", "France_bn:00036202n", "Council"], synaware_w2v=True)
