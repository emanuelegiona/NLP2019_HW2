from argparse import ArgumentParser
from nltk.corpus import wordnet as wn
from gensim.models import keyedvectors
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import spearmanr, pearsonr

import utils as u

# --- global variables ---
EMBEDDING_SIZE = 64
# --- --- ---


def score(resource_path, embeddings_path, simtest_path, use_UNK=False):
    """
    Compares scores contained in a test set with the maximum cosine similarity between all pairs of vectors associated
    to two given words' synsets, by computing Spearman and Pearson coefficients.
    :param resource_path: Path to the resource folder
    :param embeddings_path: Path to the embeddings.vec file
    :param simtest_path: Path to the similarity test file
    :param use_UNK: unknown synset vector -> True: uses vectors associated to <UNK>; False: 0-valued vector (default)
    :return: None
    """

    print("Loading vocabularies...")
    word_to_ix, _, _ = u.get_vocab(vocab_path=resource_path + "/vocab.txt",
                                   antivocab_path=resource_path + "/antivocab.txt")

    print("Loading mappings...")
    mapping, reverse_mapping = u.get_WN_mappings(resource_path + "/bn2wn_mapping.txt", with_reverse=True)

    print("Loading embeddings...")
    embeddings = keyedvectors.KeyedVectors.load_word2vec_format(embeddings_path, binary=False)

    print("Computing similarities...")
    gold_scores = []
    scores = []
    for word_pair in u.wordsim_pairs_generator(simtest_path):
        curr_score = -1
        for syn1 in wn.synsets(word_pair.word1):
            wn_id1 = "%d%s" % (syn1.offset(), syn1.pos())
            bn_id1 = reverse_mapping.get(wn_id1, None)

            # 0-valued vector in case no BabelNet ID is found
            vector1 = np.zeros(shape=[EMBEDDING_SIZE], dtype=np.float)

            if bn_id1 is None:
                if use_UNK:
                    vector1 = embeddings.get_vector("<UNK>")
            else:
                for word in word_to_ix.keys():
                    if word.find(bn_id1) > 0:
                        vector1 = embeddings.get_vector(word)
                        break

            for syn2 in wn.synsets(word_pair.word2):
                wn_id2 = "%d%s" % (syn2.offset(), syn2.pos())
                bn_id2 = reverse_mapping.get(wn_id2, None)

                # 0-valued vector in case no BabelNet ID is found
                vector2 = np.zeros(shape=[EMBEDDING_SIZE], dtype=np.float)

                if bn_id2 is None:
                    if use_UNK:
                        vector2 = embeddings.get_vector("<UNK>")
                else:
                    for word in word_to_ix.keys():
                        if word.find(bn_id2) > 0:
                            vector2 = embeddings.get_vector(word)
                            break

                cos_sim = 1.0 - cosine_dist(vector1, vector2)
                curr_score = max(curr_score, cos_sim)

        gold_scores.append(word_pair.score)
        scores.append(curr_score)

    # compute spearman and pearson coefficients
    print("\nSpearman: %.3f\nPearson: %.3f" % (spearmanr(gold_scores, scores)[0], pearsonr(gold_scores, scores)[0]))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_path", help="Path to the resource folder")
    parser.add_argument("embeddings_path", help="Path to the Path to the embeddings.vec file")
    parser.add_argument("simtest_path", help="Path to the similarity test file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score(resource_path=args.resource_path,
          embeddings_path=args.embeddings_path,
          simtest_path=args.simtest_path)
