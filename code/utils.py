from datetime import datetime
from nltk.corpus import wordnet as wn
from collections import namedtuple
import math
import random

import xmlhandler

# --- global variables ---
BATCH_SIZE_FILE = 128
PUNCTUATION_STRING = ",;.:!\"£$%&/()='?^’‘-–_"
# --- --- ---


def get_time():
    """
    Gets the current time
    :return: Current time, as String
    """

    return str(datetime.now())


def log_message(file_handle, message, to_stdout=True, with_time=True):
    """
    Log utility function
    :param file_handle: Open file handle to the log file
    :param message: Log message
    :param to_stdout: True: also prints the message to the standard output (default); False: only writes to file
    :param with_time: True: appends time at the end of the line (default); False: only prints the given message
    :return: None
    """

    if file_handle is not None:
        if with_time:
            message = "%s [%s]" % (message, get_time())
        file_handle.write("%s\n" % message)
        file_handle.flush()

    if to_stdout:
        print("%s" % message)


def get_WN_mappings(mapping_file, with_reverse=False):
    """
    Reads a file mapping BabelNet to WordNet IDs and returns a dictionary holding such mapping.
    :param mapping_file: Path to the mapping file
    :param with_reverse: True: also builds a reverse dictionary (WN2BN); False: only builds the BN2WN dictionary (default)
    :return: mapping from BabelNet IDs to WordNet IDs, as a dictionary; values are lists of WordNet IDs
    """

    dictionary = {}
    reverse_dictionary = {}

    with open(mapping_file) as file:
        for line in file:
            line = line.strip("\n").split("\t")

            if line[0] not in dictionary:
                dictionary[line[0]] = line[1:]

            if with_reverse:
                for wn_id in line[1:]:
                    if wn_id not in reverse_dictionary:
                        reverse_dictionary[wn_id] = line[0]

    if with_reverse:
        return dictionary, reverse_dictionary

    return dictionary


def get_WN_synset(wordnet_ID):
    """
    Retrieves a WordNet synset by its POS tag and offset.
    :param wordnet_ID: WordNet ID, as String
    :return: WordNet synset
    """

    pos_tag = wordnet_ID[-1]
    offset = int(wordnet_ID[:-1])
    return wn.synset_from_pos_and_offset(pos_tag, offset)


def wordsim_pairs_generator(similarity_file):
    """
    Reads a word similarity file, in which two words are annotated with a relatedness score.
    :param similarity_file: Path to the word similarity file
    :return: Pair, a namedtuple containing indices word1, word2, score, in a generator fashion
    """

    Pair = namedtuple("Pair", "word1 word2 score")

    with open(similarity_file) as file:
        header = True
        for line in file:
            if header:
                header = False
                continue

            line = line.strip("\n").split("\t")
            if len(line) != 3:
                continue

            yield Pair(word1=line[0], word2=line[1], score=float(line[2]))


def make_vocab(dataset, mapping, vocab_path, antivocab_path, min_count=5, subsampling=1e-4, logfile=None):
    """
    Creates a vocabulary of the given size from the given dataset, sorting words by frequency.
    :param dataset: File path to an XML file in EuroSense format to build the vocaqbulary of
    :param mapping: Dictionary mapping BabelNet IDs to WordNet IDs
    :param vocab_path: File path where to export the vocabulary to
    :param antivocab_path: File path where to export subsampled words to, in order not to consider them during training
    :param min_count: Minimum number of occurrences for words to be considered (default: 5)
    :param subsampling: Subsampling factor for frequent words (default: 10e-3)
    :param logfile: File handle for logging purposes
    :return: None
    """

    vocab_batch = []
    occurrences = {}
    anti_vocab = set()

    # Overwrite with fresh file, if already existing
    with open(vocab_path, mode='w', encoding='utf-8'):
        pass
    with open(antivocab_path, mode='w', encoding='utf-8'):
        pass

    with \
            open(vocab_path, mode='a', encoding='utf-8') as output, \
            open(antivocab_path, mode='a', encoding='utf-8') as antioutput, \
            xmlhandler.XmlParser(dataset, compressed=False) as parser:

        log_message(logfile, "Started vocabulary creation")
        log_message(logfile, "Started vocabulary indexing")

        count = 0
        total_words = 0
        for sentence in parser.parse(languages=["en"], replace=True, bn_to_wn_mapping=mapping):
            count += 1
            sentence = sentence[0] if len(sentence) > 0 else None
            if sentence is None:
                continue

            sentence = sentence.strip().split(" ")
            for w in sentence:

                # do not consider punctuation
                if w not in PUNCTUATION_STRING:
                    occurrences[w] = occurrences.get(w, 0) + 1
                    total_words += 1

            if count % 50_000 == 0:
                log_message(logfile, "%d sentences parsed; %d words so far" % (count, len(occurrences)))

        log_message(logfile, "Finished vocabulary indexing; %d total sentences; %d total words" % (count, len(occurrences)))

        # sort by decreasing occurrence number to only get the most frequent words
        for w in sorted(occurrences.items(), key=lambda x: x[1], reverse=True):

            # lemma_synset like words have always to be kept
            has_synset = w[0].find("_bn:") > 0

            # only consider words with at least min_count occurrences
            if has_synset or w[1] >= min_count:

                # apply subsampling
                prob = 1.0 - math.sqrt(subsampling / w[1] * total_words)
                if has_synset or random.uniform(0, 1) >= prob:
                    # take the word
                    vocab_batch.append(w[0])
                else:
                    anti_vocab.add(w[0])

            # BATCH_SIZE reached
            if len(vocab_batch) == BATCH_SIZE_FILE:
                output.writelines("%s\n" % v for v in vocab_batch)
                vocab_batch = []

        del occurrences

        # incomplete batch
        if len(vocab_batch) > 0:
            output.writelines("%s\n" % v for v in vocab_batch)
            del vocab_batch

        # export too frequent words
        antioutput.writelines("%s\n" % a for a in anti_vocab)

    log_message(logfile, "Finished vocabulary creation")


def get_vocab(vocab_path, antivocab_path):
    """
    Reads vocabulary and antivocabulary (subsampled words) from their respective files.
    :param vocab_path: File path to vocabulary
    :param antivocab_path: File path to antivocabulary
    :return: (word to index as Dict, index to word as List, subsampled_words as List)
    """

    word_to_ix = {"<PAD>": 0, "<UNK>": 1}
    ix_to_word = ["<PAD>", "<UNK>"]
    subsampled_words = set()

    with open(vocab_path, encoding="utf-8") as file:
        for word in file:
            word = word.strip()

            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
                ix_to_word.append(word)

    with open(antivocab_path, encoding="utf-8") as file:
        for word in file:
            subsampled_words.add(word)

    return word_to_ix, ix_to_word, list(subsampled_words)


def make_dataset(dataset, mapping, output):
    """
    Converts an XML file in the EuroSense format to a plain sentence representation.
    :param dataset: File path to an XML file in EuroSense format to build the vocaqbulary of
    :param mapping: Dictionary mapping BabelNet IDs to WordNet IDs
    :param output: File path to the output file
    :return: None
    """

    with open(output, mode="w", encoding="utf-8"):
        pass

    with \
            open(output, mode="a", encoding="utf-8") as output_file, \
            xmlhandler.XmlParser(dataset, compressed=False) as parser:

        output_batch = []
        for sentence in parser.parse(languages=["en"], replace=True, bn_to_wn_mapping=mapping):
            sentence = sentence[0] if len(sentence) > 0 else None
            if sentence is None:
                continue

            sentence = sentence.strip()

            output_batch.append(sentence)
            if len(output_batch) == BATCH_SIZE_FILE:
                output_file.writelines("%s\n" % s for s in output_batch)
                output_batch = []

        if len(output_batch) > 0:
            output_file.writelines("%s\n" % s for s in output_batch)
            del output_batch


def get_synset_vocab(word_to_ix):
    """
    Given a vocabulary, builds a same synset vocabulary.
    :param word_to_ix: Vocabulary obtained from utils.get_vocab, as Dict
    :return: Synset vocabulary mapping a BabelNet synset ID to a List of integers consistent with word_to_ix, as Dict
    """

    syn_to_ix = {}

    for word in word_to_ix.keys():
        bn_id_pos = word.find("_bn:")
        if bn_id_pos < 0:
            continue

        bn_id = word[bn_id_pos + 1:]
        if bn_id not in syn_to_ix:
            syn_to_ix[bn_id] = [word_to_ix[word]]
        else:
            syn_to_ix[bn_id].append(word_to_ix[word])

    return syn_to_ix


if __name__ == "__main__":
    #mapping = get_WN_mappings("../resources/bn2wn_mapping.txt")

    #with open("../logs/vocab.log", "w") as vocab_log:
    #    make_vocab(dataset="../resources/eurosense.v1.0.english_only.xml",
    #               mapping=mapping,
    #               vocab_path="../resources/vocab.txt",
    #               antivocab_path="../resources/antivocab.txt",
    #               logfile=vocab_log)

    #make_dataset("../resources/eurosense.v1.0.english_only.xml", mapping, "../resources/eurosense_sentences.txt")

    word_to_ix, ix_to_word, _ = get_vocab("../resources/vocab.txt", "../resources/antivocab.txt")
    syn_to_ix = get_synset_vocab(word_to_ix)
    print(len(syn_to_ix))
    for syn, words in syn_to_ix.items():
        if len(words) > 1:
            print(syn)
            print(random.sample([ix_to_word[w] for w in words], k=3))
            break
