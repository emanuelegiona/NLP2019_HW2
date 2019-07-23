import tensorflow as tf

import utils as u
from model import Word2Vec, SynsetAwareWord2Vec


# --- global variables ---
SAVE_FREQUENCY = 5
# --- --- ---


def add_summary(writer, name, value, global_step):
    """
    Utility function to track the model's progess in TensorBoard
    :param writer: tf.summary.FileWriter instance
    :param name: Value label to be shown in TensorBoard
    :param value: Value to append for the current step
    :param global_step: Current step for which the value has to be considered
    :return: None
    """

    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
    writer.add_summary(summary, global_step=global_step)


def train_basic_w2v(dataset,
                    word_to_ix,
                    subsampled_words,
                    model_path,
                    model_ID,
                    epochs,
                    batch_size,
                    embedding_size,
                    lr,
                    window_size,
                    neg_samples,
                    retrain=False,
                    csv_export=True,
                    ix_to_word=None,
                    emb_export=False,
                    emb_path=None):
    """
    Trains a basic Word2Vec model with the given parameters, exporting embeddings in the end.
    :param dataset: Path to the dataset
    :param word_to_ix: Vocabulary to map words to integer IDs, as Dict
    :param subsampled_words: Words not to be considered since too frequent, as List
    :param model_path: Path to the outermost model folder
    :param model_ID: Name of the model being trained
    :param epochs: Number of epochs to use for training
    :param batch_size: Size of batches to be built
    :param embedding_size: Size of embeddings to be built
    :param lr: Learning rate for the training algorithm
    :param window_size: Number of words to consider as context; the full context is 2 * window_size large
    :param neg_samples: Number of negative samples to be used in the Noise Contrastive Estimation loss function
    :param retrain: True: loads the model and starts training again; False: starts a new training (default)
    :param csv_export: True: exports train and dev loss values per epoch to a CSV file (default); False: nothing
    :param ix_to_word: Reverse vocabulary (wrt to word_to_ix); MUST be provided if embeddings are to be exported
    :param emb_export: True: exports embeddings to a W2V textual format; False: nothing
    :param emb_path: Path to the embedding export file; MUST be provided if embeddings are to be exported
    :return: None
    """

    assert not emb_export or (emb_export and ix_to_word is not None and emb_path is not None), "Embeddings export enabled but no reverse dictionary or embedding file path provided"

    with \
            tf.Session() as sess, \
            tf.summary.FileWriter("../logging/%s" % model_ID, sess.graph) as tf_logger, \
            open("../logs/training_%s.log" % model_ID, mode="w") as log:

        if csv_export:
            csv = open("../csv/%s.csv" % model_ID, mode="w")
            u.log_message(csv, "epoch,train loss,dev loss", to_stdout=False, with_time=False)

        u.log_message(log, "Creating model...")
        model = Word2Vec(subsampled_words=subsampled_words,
                         vocabulary_size=len(word_to_ix),
                         embedding_size=embedding_size,
                         learning_rate=lr,
                         window_size=window_size,
                         neg_samples=neg_samples)

        u.log_message(log, "\tModel ID: %s" % model_ID)
        u.log_message(log, "\tModel path: %s/%s/model.ckpt" % (model_path, model_ID))
        u.log_message(log, "\tEmbedding size: %d" % embedding_size)
        u.log_message(log, "\tLearning rate: %.3f" % lr)
        u.log_message(log, "\tWindow size: %d" % window_size)
        u.log_message(log, "\tNegative samples: %d" % neg_samples)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        if retrain:
            u.log_message(log, "Loading model...")
            saver.restore(sess, "%s/%s/model.ckpt" % (model_path, model_ID))

        u.log_message(log, "Starting training...")
        for epoch in range(1, epochs+1):
            u.log_message(log, "Epoch: %d" % epoch)
            accumulated_loss = 0
            iterations = 0

            # training
            for batch_inputs, batch_labels in model.batch_generator(dataset=dataset,
                                                                    word_to_ix=word_to_ix,
                                                                    batch_size=batch_size,
                                                                    sent_limit=1720588):    # around 90% of 1_911_765

                _, loss_val = sess.run([model.train, model.loss],
                                       feed_dict={model.data["inputs"]: batch_inputs,
                                                  model.data["labels"]: batch_labels})

                accumulated_loss += loss_val
                iterations += 1

            accumulated_loss /= iterations
            train_loss = accumulated_loss

            u.log_message(log, "\tTrain loss: %.5f" % train_loss)
            add_summary(tf_logger,
                        "train loss",
                        train_loss,
                        epoch)

            # dev evaluation
            accumulated_loss = 0
            iterations = 0

            for batch_inputs, batch_labels in model.batch_generator(dataset=dataset,
                                                                    word_to_ix=word_to_ix,
                                                                    batch_size=batch_size,
                                                                    skip_first=1720589):    # use the remaining 191_177 as dev set

                loss_val = sess.run(model.loss,
                                    feed_dict={model.data["inputs"]: batch_inputs,
                                               model.data["labels"]: batch_labels})

                accumulated_loss += loss_val
                iterations += 1

            accumulated_loss /= iterations
            dev_loss = accumulated_loss

            u.log_message(log, "\tDev loss: %.5f" % dev_loss)
            add_summary(tf_logger,
                        "dev loss",
                        dev_loss,
                        epoch)

            if csv_export:
                u.log_message(csv, "%d,%.5f,%.5f" % (epoch, train_loss, dev_loss), to_stdout=False, with_time=False)

            if epoch % SAVE_FREQUENCY == 0:
                saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))
                u.log_message(log, "\tModel saved")

        u.log_message(log, "Training ended.")
        saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))

        if emb_export:
            u.log_message(log, "Exporting embeddings...")
            emb_matrix = sess.run(model.embeddings)
            model.export_keyedvector(emb_matrix, embeddings_path=emb_path, ix_to_word=ix_to_word)
            u.log_message(log, "Embeddings exported")

        if csv_export:
            csv.close()


def train_synaware_w2v(dataset,
                       word_to_ix,
                       syn_to_ix,
                       subsampled_words,
                       model_path,
                       model_ID,
                       epochs,
                       batch_size,
                       embedding_size,
                       lr,
                       window_size,
                       neg_samples,
                       sample_k=3,
                       retrain=False,
                       csv_export=True,
                       ix_to_word=None,
                       emb_export=False,
                       emb_path=None):
    """
    Trains a synset-aware Word2Vec model with the given parameters, exporting embeddings in the end.
    :param dataset: Path to the dataset
    :param word_to_ix: Vocabulary to map words to integer IDs, as Dict
    :param syn_to_ix: Mapping from BabelNet synset IDs to integers, as Dict
    :param subsampled_words: Words not to be considered since too frequent, as List
    :param model_path: Path to the outermost model folder
    :param model_ID: Name of the model being trained
    :param epochs: Number of epochs to use for training
    :param batch_size: Size of batches to be built
    :param embedding_size: Size of embeddings to be built
    :param lr: Learning rate for the training algorithm
    :param window_size: Number of words to consider as context; the full context is 2 * window_size large
    :param neg_samples: Number of negative samples to be used in the Noise Contrastive Estimation loss function
    :param sample_k: Number of words sharing the same synset to sample (default: 3)
    :param retrain: True: loads the model and starts training again; False: starts a new training (default)
    :param csv_export: True: exports train and dev loss values per epoch to a CSV file (default); False: nothing
    :param ix_to_word: Reverse vocabulary (wrt to word_to_ix); MUST be provided if embeddings are to be exported
    :param emb_export: True: exports embeddings to a W2V textual format; False: nothing
    :param emb_path: Path to the embedding export file; MUST be provided if embeddings are to be exported
    :return: None
    """

    assert not emb_export or (emb_export and ix_to_word is not None and emb_path is not None), "Embeddings export enabled but no reverse dictionary or embedding file path provided"

    with \
            tf.Session() as sess, \
            tf.summary.FileWriter("../logging/%s" % model_ID, sess.graph) as tf_logger, \
            open("../logs/training_%s.log" % model_ID, mode="w") as log:

        if csv_export:
            csv = open("../csv/%s.csv" % model_ID, mode="w")
            u.log_message(csv, "epoch,train loss,dev loss", to_stdout=False, with_time=False)

        u.log_message(log, "Creating model...")
        model = SynsetAwareWord2Vec(subsampled_words=subsampled_words,
                                    vocabulary_size=len(word_to_ix),
                                    embedding_size=embedding_size,
                                    learning_rate=lr,
                                    window_size=window_size,
                                    neg_samples=neg_samples)

        u.log_message(log, "\tModel ID: %s" % model_ID)
        u.log_message(log, "\tModel path: %s/%s/model.ckpt" % (model_path, model_ID))
        u.log_message(log, "\tEmbedding size: %d" % embedding_size)
        u.log_message(log, "\tLearning rate: %.3f" % lr)
        u.log_message(log, "\tWindow size: %d" % window_size)
        u.log_message(log, "\tNegative samples: %d" % neg_samples)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        if retrain:
            u.log_message(log, "Loading model...")
            saver.restore(sess, "%s/%s/model.ckpt" % (model_path, model_ID))

        u.log_message(log, "Starting training...")
        for epoch in range(1, epochs+1):
            u.log_message(log, "Epoch: %d" % epoch)
            accumulated_loss = 0
            iterations = 0

            # training
            for batch_inputs, batch_labels, batch_same_synsets in model.batch_generator(dataset=dataset,
                                                                                        word_to_ix=word_to_ix,
                                                                                        syn_to_ix=syn_to_ix,
                                                                                        batch_size=batch_size,
                                                                                        sample_k=sample_k,
                                                                                        sent_limit=1720588):    # around 90% of 1_911_765

                model.batch_size = len(batch_inputs)
                _, loss_val = sess.run([model.train, model.loss],
                                       feed_dict={model.data["inputs"]: batch_inputs,
                                                  model.data["labels"]: batch_labels,
                                                  model.data["same_synset"]: batch_same_synsets})

                accumulated_loss += loss_val
                iterations += 1

            accumulated_loss /= iterations
            train_loss = accumulated_loss

            u.log_message(log, "\tTrain loss: %.5f" % train_loss)
            add_summary(tf_logger,
                        "train loss",
                        train_loss,
                        epoch)

            # dev evaluation
            accumulated_loss = 0
            iterations = 0

            for batch_inputs, batch_labels, batch_same_synsets in model.batch_generator(dataset=dataset,
                                                                                        word_to_ix=word_to_ix,
                                                                                        syn_to_ix=syn_to_ix,
                                                                                        batch_size=batch_size,
                                                                                        skip_first=1720589):    # use the remaining 191_177 as dev set

                loss_val = sess.run(model.loss,
                                    feed_dict={model.data["inputs"]: batch_inputs,
                                               model.data["labels"]: batch_labels,
                                               model.data["same_synset"]: batch_same_synsets})

                accumulated_loss += loss_val
                iterations += 1

            accumulated_loss /= iterations
            dev_loss = accumulated_loss

            u.log_message(log, "\tDev loss: %.5f" % dev_loss)
            add_summary(tf_logger,
                        "dev loss",
                        dev_loss,
                        epoch)

            if csv_export:
                u.log_message(csv, "%d,%.5f,%.5f" % (epoch, train_loss, dev_loss), to_stdout=False, with_time=False)

            if epoch % SAVE_FREQUENCY == 0:
                saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))
                u.log_message(log, "\tModel saved")

        u.log_message(log, "Training ended.")
        saver.save(sess, "%s/%s/model.ckpt" % (model_path, model_ID))

        if emb_export:
            u.log_message(log, "Exporting embeddings...")
            emb_matrix = sess.run(model.embeddings)
            model.export_keyedvector(emb_matrix, embeddings_path=emb_path, ix_to_word=ix_to_word)
            u.log_message(log, "Embeddings exported")

        if csv_export:
            csv.close()


def grid_search():
    embedding_sizes = [32, 64]
    learning_rates = [0.15, 0.10]
    window_sizes = [2, 3]

    word_to_ix, ix_to_word, subsampled_words = u.get_vocab(vocab_path="../resources/vocab.txt",
                                                           antivocab_path="../resources/antivocab.txt")

    for e_size in embedding_sizes:
        for lr in learning_rates:
            for w_size in window_sizes:
                tf.reset_default_graph()

                train_basic_w2v(dataset="../resources/eurosense_sentences.txt",
                                word_to_ix=word_to_ix,
                                subsampled_words=subsampled_words,
                                model_path="../resources/models",
                                model_ID="basic_w2v_E%d_LR%.3f_W%d" % (e_size, lr, w_size),
                                epochs=30,
                                batch_size=64,
                                embedding_size=e_size,
                                lr=lr,
                                window_size=w_size,
                                neg_samples=16,
                                csv_export=False)


if __name__ == "__main__":
    word_to_ix, ix_to_word, subsampled_words = u.get_vocab(vocab_path="../resources/vocab.txt",
                                                           antivocab_path="../resources/antivocab.txt")

    syn_to_ix = u.get_synset_vocab(word_to_ix)

    #grid_search()

    tf.reset_default_graph()
    #train_basic_w2v(dataset="../resources/eurosense_sentences.txt",
    #                word_to_ix=word_to_ix,
    #                subsampled_words=subsampled_words,
    #                model_path="../resources/models",
    #                model_ID="basic_w2v",
    #                epochs=30,
    #                batch_size=64,
    #                embedding_size=64,
    #                lr=0.15,
    #                window_size=3,
    #                neg_samples=16,
    #                ix_to_word=ix_to_word,
    #                emb_export=True,
    #                emb_path="../resources/embeddings.vec")

    train_synaware_w2v(dataset="../resources/eurosense_sentences.txt",
                       word_to_ix=word_to_ix,
                       syn_to_ix=syn_to_ix,
                       subsampled_words=subsampled_words,
                       model_path="../resources/models",
                       model_ID="synaware_w2v",
                       epochs=30,
                       batch_size=64,
                       embedding_size=64,
                       lr=0.15,
                       window_size=3,
                       neg_samples=16,
                       ix_to_word=ix_to_word,
                       emb_export=True,
                       emb_path="../resources/embeddings_synaware_w2v.vec")
