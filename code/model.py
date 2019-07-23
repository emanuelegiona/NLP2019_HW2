import functools
import numpy as np
import tensorflow as tf
import math
from collections import namedtuple
import random

import utils as u


# only evalutate a function once, returning a member holding the instance for future calls
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# --- --- ---


class Word2Vec:
    """
    Basic Word2Vec implementation.
    """

    def __init__(self, subsampled_words, vocabulary_size, embedding_size, learning_rate, window_size, neg_samples):
        """
        Initializes a Word2Vec model with the given parameters.
        :param subsampled_words: words to be excluded from the window in a CBOW implementation, as List
        :param vocabulary_size: Size of the vocabulary
        :param embedding_size: Size of the embeddings to create
        :param learning_rate: Learning rate for the Adagrad optimizer
        :param window_size: Size of the window in a CBOW implementation
        :param neg_samples: Number of negative samples to be used in a NCE loss function
        """

        self.subsampled_words = subsampled_words
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.neg_samples = neg_samples

        # correctly initialize graph
        self.data
        self.embeddings
        self.lookups
        self.dense
        self.loss
        self.train
        self.similarity

    @lazy_property
    def data(self):
        with tf.variable_scope("inputs"):
            inputs = tf.placeholder(tf.int32, name="inputs", shape=[None, 2 * self.window_size])
            labels = tf.placeholder(tf.int32, name="labels", shape=[None, 1])
            sim_test = tf.placeholder(tf.int32, name="sim_test", shape=[None])
            return {"inputs": inputs, "labels": labels, "sim_test": sim_test}

    @lazy_property
    def embeddings(self):
        with tf.variable_scope("embeddings"):
            return tf.get_variable(name="embeddings",
                                   initializer=tf.random_uniform(
                                       [self.vocabulary_size, self.embedding_size],
                                       -1.0,
                                       1.0),
                                   dtype=tf.float32)

    @lazy_property
    def lookups(self):
        lookups = tf.nn.embedding_lookup(self.embeddings, self.data["inputs"])
        return tf.reduce_mean(lookups, axis=1)

    @lazy_property
    def dense(self):
        Dense = namedtuple("Dense", "weights bias")
        with tf.variable_scope("dense"):
            weights = tf.get_variable(name="W",
                                      initializer=tf.truncated_normal(
                                          [self.vocabulary_size, self.embedding_size],
                                          stddev=1.0 / math.sqrt(self.embedding_size))
                                      )

            bias = tf.get_variable(name="bias",
                                   initializer=tf.zeros(self.vocabulary_size))
            return Dense(weights, bias)

    @lazy_property
    def loss(self):
        with tf.variable_scope("loss"):
            return tf.reduce_mean(tf.nn.nce_loss(weights=self.dense.weights,
                                                 biases=self.dense.bias,
                                                 inputs=self.lookups,
                                                 labels=self.data["labels"],
                                                 num_sampled=self.neg_samples,
                                                 num_classes=self.vocabulary_size)
                                  )

    @lazy_property
    def train(self):
        with tf.variable_scope("train"):
            return tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    @lazy_property
    def similarity(self):
        with tf.variable_scope("similarity"):
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            normalized_embeddings = self.embeddings / norm
            test_lookups = tf.nn.embedding_lookup(normalized_embeddings, self.data["sim_test"])
            return tf.matmul(test_lookups, normalized_embeddings, transpose_b=True)

    def prepare_sentence(self, sentence, word_to_ix):
        """
        Constructs a CBOW input record for the given sentence.
        :param sentence: Input sentence
        :param word_to_ix: Mapping from vocabulary to integers, as a Dict
        :return: tuples of shape (context, label) in a generator fashion
        """

        sentence = sentence.strip("\n").split(" ")

        # remove punctuation
        sentence = [w for w in sentence if w not in u.PUNCTUATION_STRING]

        # remove subsampled words
        sentence = [w for w in sentence if w not in self.subsampled_words]

        # replace with the appropriate word ID or UNK's ID if the word is not present in the vocabulary
        sentence = [word_to_ix[w] if w in word_to_ix else word_to_ix["<UNK>"] for w in sentence]

        sentence_size = len(sentence)
        for label_id in range(0, sentence_size):
            lower_bound = label_id - self.window_size
            upper_bound = label_id + self.window_size + 1

            min_id = max(0, lower_bound)
            max_id = min(sentence_size, upper_bound)

            context = [sentence[x] for x in range(min_id, max_id) if x != label_id]

            # "-lower_bound" = number of missing words in the left side of the context
            if lower_bound < 0:
                context = ([word_to_ix["<PAD>"]] * (-lower_bound)) + context

            # upper_bound - sentence_size = number of missing words in the right side of the context
            if upper_bound > sentence_size:
                context = context + ([word_to_ix["<PAD>"]] * (upper_bound - sentence_size))

            label = sentence[label_id]

            yield context, label

    def batch_generator(self, dataset, word_to_ix, batch_size, shuffle=True, skip_first=0, sent_limit=-1):
        """
        Builds batches of a given size out of the dataset, shuffling them if needed.
        :param dataset: File path to the dataset
        :param word_to_ix: Mapping from vocabulary to integers, as a Dict
        :param batch_size: Size of batches
        :param shuffle: True: shuffles the batch (default)
        :param skip_first: Number of sentences to skip from the start of the dataset. If 0: no sentence skipped (default)
        :param sent_limit: Number of sentences to extract from the dataset. If -1: reads all dataset (default)
        :return: tuples of shape (batch_inputs, batch_labels) in a generator fashion
        """

        with open(dataset, encoding="utf-8") as file:
            batch_inputs = np.zeros(shape=(batch_size, 2 * self.window_size), dtype=np.int32)
            batch_labels = np.zeros(shape=(batch_size, 1), dtype=np.int32)
            curr_size = 0
            sent_number = 0

            for sentence in file:
                if len(sentence.split(" ")) == 0:
                    continue

                if skip_first > 0:
                    skip_first -= 1
                    continue

                if sent_limit > 0 and sent_number == sent_limit:
                    break
                sent_number += 1

                for context, label in self.prepare_sentence(sentence, word_to_ix):
                    batch_inputs[curr_size, :] = [w for w in context]
                    batch_labels[curr_size, 0] = label
                    curr_size += 1

                    if curr_size == batch_size:
                        if shuffle:
                            perm = np.random.permutation(batch_size)
                            batch_inputs = batch_inputs[perm]
                            batch_labels = batch_labels[perm]

                        yield batch_inputs, batch_labels

                        batch_inputs = np.zeros(shape=(batch_size, 2 * self.window_size), dtype=np.int32)
                        batch_labels = np.zeros(shape=(batch_size, 1), dtype=np.int32)
                        curr_size = 0

            if curr_size > 0:
                if shuffle:
                    perm = np.random.permutation(curr_size)
                    batch_inputs = batch_inputs[perm]
                    batch_labels = batch_labels[perm]

                yield batch_inputs[:curr_size], batch_labels[:curr_size]

    def export_keyedvector(self, emb_matrix, embeddings_path, ix_to_word, with_UNK=False):
        """
        Exports the embedding matrix to file, compliant to the Word2Vec textual format.
        :param emb_matrix: Initialized embedding matrix, obtained running the TensorFlow operation "self.embeddings"
        :param embeddings_path: File path to the embeddings file to be written
        :param ix_to_word: Reverse vocabulary, mapping integer IDs to words, as List
        :param with_UNK: True: also includes the <UNK> entry for the embedding matrix; False: only lemma_synset entries are kept (default)
        :return: None
        """

        KeyedVector = namedtuple("KeyedVector", "key vector")
        vectors = []
        for i in range(self.vocabulary_size):
            word = ix_to_word[i]

            # only keep embeddings for lemma_synset like words or UNK (if specified)
            if word.find("_bn:") > 0 or (with_UNK and word == "<UNK>"):
                vector = emb_matrix[i, :]
                vectors.append(KeyedVector(key=word,
                                           vector=(" ".join(map(str, vector)))))

        with open(embeddings_path, encoding="utf-8", mode="w") as file:
            file.write("%d %d\n" % (len(vectors), self.embedding_size))
            file.flush()

            file.writelines("%s %s\n" % (v.key, v.vector) for v in vectors)
            file.flush()


class SynsetAwareWord2Vec(Word2Vec):
    """
    Synset aware Word2Vec implementation.
    Works like a simple Word2Vec model, but the loss also takes account for other words which refer to the same
    synset as the labels'.
    This should force the model to cluster vector values for different words sharing the same synset.
    """

    def __init__(self, subsampled_words, vocabulary_size, embedding_size, learning_rate, window_size, neg_samples):
        """
        Initializes a SynsetAwareWord2Vec model with the given parameters.
        :param subsampled_words: words to be excluded from the window in a CBOW implementation, as List
        :param vocabulary_size: Size of the vocabulary
        :param embedding_size: Size of the embeddings to create
        :param learning_rate: Learning rate for the Adagrad optimizer
        :param window_size: Size of the window in a CBOW implementation
        :param neg_samples: Number of negative samples to be used in a NCE loss function
        """

        self.subsampled_words = subsampled_words
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.neg_samples = neg_samples

        # correctly initialize graph
        self.data
        self.embeddings
        self.lookups
        self.dense
        self.loss
        self.train
        self.similarity

    @lazy_property
    def data(self):
        with tf.variable_scope("inputs"):
            inputs = tf.placeholder(tf.int32, name="inputs", shape=[None, 2 * self.window_size])
            labels = tf.placeholder(tf.int32, name="labels", shape=[None, 1])
            same_synset = tf.placeholder(tf.int32, name="same_synset", shape=[None, None])  # other words sharing the same synset as label

            sim_test = tf.placeholder(tf.int32, name="sim_test", shape=[None])
            return {"inputs": inputs,
                    "labels": labels,
                    "same_synset": same_synset,
                    "sim_test": sim_test}

    @lazy_property
    def loss(self):
        with tf.variable_scope("loss"):
            nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.dense.weights,
                                                     biases=self.dense.bias,
                                                     inputs=self.lookups,
                                                     labels=self.data["labels"],
                                                     num_sampled=self.neg_samples,
                                                     num_classes=self.vocabulary_size)
                                      )

            # compute whether to take into account the same synset word or not based on value (0 = PAD token)
            same_synset_length = tf.shape(self.data["same_synset"])[1]
            mean_weights = tf.not_equal(self.data["same_synset"], tf.constant(0, dtype=tf.int32))
            mean_weights = tf.cast(mean_weights, tf.int32)
            mean_weights = tf.reshape(mean_weights, shape=[-1, same_synset_length, 1])

            # retrieve embeddings for both labels and their associated same synset words
            label_embs = tf.nn.embedding_lookup(self.embeddings, self.data["labels"])
            same_synset_embs = tf.nn.embedding_lookup(self.embeddings, self.data["same_synset"])

            # prepare for cosine distance
            label_embs = tf.tile(input=label_embs, multiples=[1, same_synset_length, 1])
            label_embs = tf.nn.l2_normalize(label_embs, axis=-1)
            same_synset_embs = tf.nn.l2_normalize(same_synset_embs, axis=-1)

            # force label vector to be close to other vectors for words sharing the same synset
            mean_cosine_distance = tf.losses.cosine_distance(label_embs,
                                                             same_synset_embs,
                                                             axis=-1,
                                                             weights=mean_weights)

            return nce_loss + mean_cosine_distance

    def prepare_sentence(self, sentence, word_to_ix, syn_to_ix, sample_k):
        """
        Constructs a CBOW input record augmented with words sharing the same synset of the label for the given sentence.
        :param sentence: Input sentence
        :param word_to_ix: Mapping from vocabulary to integers, as a Dict
        :param syn_to_ix: Mapping from BabelNet synset IDs to integers, as Dict
        :param sample_k: Number of words sharing the same synset to sample
        :return: tuples of shape (context, label, same_synset_words) in a generator fashion
        """

        sentence = sentence.strip("\n").split(" ")

        # remove punctuation
        sentence = [w for w in sentence if w not in u.PUNCTUATION_STRING]

        # remove subsampled words
        sentence = [w for w in sentence if w not in self.subsampled_words]

        # retrieve BabelNet IDs for all words in the sentence
        fetch_id = lambda w: w[w.find("_bn:") + 1:] if w.find("_bn:") > 0 else None
        bn_ids = [fetch_id(w) for w in sentence]

        # replace with the appropriate word ID or UNK's ID if the word is not present in the vocabulary
        sentence = [word_to_ix[w] if w in word_to_ix else word_to_ix["<UNK>"] for w in sentence]

        sentence_size = len(sentence)
        for label_id in range(0, sentence_size):
            lower_bound = label_id - self.window_size
            upper_bound = label_id + self.window_size + 1

            min_id = max(0, lower_bound)
            max_id = min(sentence_size, upper_bound)

            context = [sentence[x] for x in range(min_id, max_id) if x != label_id]

            # "-lower_bound" = number of missing words in the left side of the context
            if lower_bound < 0:
                context = ([word_to_ix["<PAD>"]] * (-lower_bound)) + context

            # upper_bound - sentence_size = number of missing words in the right side of the context
            if upper_bound > sentence_size:
                context = context + ([word_to_ix["<PAD>"]] * (upper_bound - sentence_size))

            label = sentence[label_id]

            # select sample_k words sharing the same synset via sampling; apply padding if needed
            bn_id = bn_ids[label_id]
            if bn_id is not None:
                same_synset_words = syn_to_ix.get(bn_id, [])
                if len(same_synset_words) > sample_k:
                    same_synset_words = random.sample(same_synset_words, k=sample_k)
                else:
                    missing = sample_k - len(same_synset_words)
                    same_synset_words += [word_to_ix["<PAD>"]] * missing
            else:
                same_synset_words = [word_to_ix["<PAD>"]] * sample_k

            yield context, label, same_synset_words

    def batch_generator(self, dataset, word_to_ix, syn_to_ix, batch_size, sample_k=3, shuffle=True, skip_first=0, sent_limit=-1):
        """
        Builds batches of a given size out of the dataset, shuffling them if needed.
        :param dataset: File path to the dataset
        :param word_to_ix: Mapping from vocabulary to integers, as a Dict
        :param syn_to_ix: Mapping from BabelNet synset IDs to integers, as Dict
        :param batch_size: Size of batches
        :param sample_k: Number of words sharing the same synset to sample (default: 3)
        :param shuffle: True: shuffles the batch (default)
        :param skip_first: Number of sentences to skip from the start of the dataset. If 0: no sentence skipped (default)
        :param sent_limit: Number of sentences to extract from the dataset. If -1: reads all dataset (default)
        :return: tuples of shape (batch_inputs, batch_labels, batch_same_synset) in a generator fashion
        """

        with open(dataset, encoding="utf-8") as file:
            batch_inputs = np.zeros(shape=(batch_size, 2 * self.window_size), dtype=np.int32)
            batch_labels = np.zeros(shape=(batch_size, 1), dtype=np.int32)
            batch_same_synsets = np.zeros(shape=(batch_size, sample_k), dtype=np.int32)
            curr_size = 0
            sent_number = 0

            for sentence in file:
                if len(sentence.split(" ")) == 0:
                    continue

                if skip_first > 0:
                    skip_first -= 1
                    continue

                if sent_limit > 0 and sent_number == sent_limit:
                    break
                sent_number += 1

                for context, label, same_synset_words in self.prepare_sentence(sentence, word_to_ix, syn_to_ix, sample_k):
                    batch_inputs[curr_size, :] = [w for w in context]
                    batch_labels[curr_size, 0] = label
                    batch_same_synsets[curr_size, :] = [w for w in same_synset_words]
                    curr_size += 1

                    if curr_size == batch_size:
                        if shuffle:
                            perm = np.random.permutation(batch_size)
                            batch_inputs = batch_inputs[perm]
                            batch_labels = batch_labels[perm]
                            batch_same_synsets = batch_same_synsets[perm]

                        yield batch_inputs, batch_labels, batch_same_synsets

                        batch_inputs = np.zeros(shape=(batch_size, 2 * self.window_size), dtype=np.int32)
                        batch_labels = np.zeros(shape=(batch_size, 1), dtype=np.int32)
                        batch_same_synsets = np.zeros(shape=(batch_size, sample_k), dtype=np.int32)
                        curr_size = 0

            if curr_size > 0:
                if shuffle:
                    perm = np.random.permutation(curr_size)
                    batch_inputs = batch_inputs[perm]
                    batch_labels = batch_labels[perm]
                    batch_same_synsets = batch_same_synsets[perm]

                yield batch_inputs[:curr_size], batch_labels[:curr_size], batch_same_synsets[:curr_size]
