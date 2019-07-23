from nltk.corpus import wordnet as wn

import utils as u


class SenseAligner:
    """
    Class used to align sense annotations in XML file which is in the EuroSense format.
    """

    def __init__(self, bn_to_wn_mapping, debug=0):
        """
        Constructor function.
        :param bn_to_wn_mapping: File path to the mapping file between BabelNet and WordNet
        :param debug: 0: no output messages (default), 1: normal debug messages, 2: verbose debug messages
        """
        self.debug = debug
        self.mapping = u.get_WN_mappings(bn_to_wn_mapping)

    def process(self, sentence, annotations_attributes, annotations):
        """
        Aligns senses for a given sentence.
        For each BabelNet ID annotation, it retrieves its associated WordNet IDs and unifies all lemmas associated to them into a bag of lemmas.
        In case the annotated lemma is not found in the bag of lemmas, the anchor is deemed incorrect.

        The word which has a synset whose lemmas intersect with the bag of lemmas previously computed is taken as the new anchor.
        If no such word is found, the annotation is discarded.

        :param sentence: Sentence to be checked for sense alignment, as String
        :param annotations_attributes: Annotation attributes for the sentence, as List of Dicts
        :param annotations: Annotations, as List of BabelNet IDs
        :return: (new annotation attributes, new annotations, number of alignments performed), both subsets of the original Lists, with 'anchor' and 'lemma' correctly aligned
        """

        if sentence is None or len(sentence) == 0:
            return [], [], 0

        new_ann_attributes = []
        new_annotations = []
        alignments = 0

        if self.debug > 0:
            print("\n--- before ---")
            for i in range(len(annotations)):
                print(i, annotations_attributes[i], annotations[i])

        for i in range(len(annotations)):
            curr_ann_attributes = annotations_attributes[i]
            curr_annotation = annotations[i]

            # early stop flag, once the correct anchor is found
            stop = False

            # get all mappings to WordNet IDs
            wn_mappings = self.mapping.get(curr_annotation, None)

            # discard if not in WordNet
            if wn_mappings is None:
                continue

            # unify all lemmas from all the WordNet IDs mapped to the same BabelNet ID
            real_lemmas = set()
            for wn_id in wn_mappings:
                real_synset = u.get_WN_synset(wn_id)

                for lemma in real_synset.lemmas():
                    real_lemmas.add(lemma.name())

            annotated_anchor = curr_ann_attributes["anchor"]
            annotated_lemma = curr_ann_attributes["lemma"]

            if self.debug > 1:
                print("\nReal lemmas: %s\n... but current lemma annotation is: %s\n\n" % (real_lemmas, annotated_lemma))

            if annotated_lemma not in real_lemmas:
                if self.debug > 0:
                    print("Invalid anchor spotted: %s" % annotated_anchor)

                # mark the ones posing problems
                curr_ann_attributes["anchor"] = "<UNDEFINED>"
                curr_ann_attributes["lemma"] = "<UNDEFINED>"

                # find the correct anchor
                for word in sentence.split(" "):

                    # it makes no sense comparing it to the wrong anchor again
                    if word == annotated_anchor:
                        continue

                    # the anchor has been found
                    if stop:
                        break

                    if self.debug > 1:
                        print("\tTesting anchor: %s\n" % word)

                    for synset in wn.synsets(word):

                        # the synset has been found
                        if stop:
                            break

                        lemmas = set([lemma.name() for lemma in synset.lemmas()])
                        intersect = real_lemmas.intersection(lemmas)

                        if self.debug > 1:
                            print("\t... has synset: %s\n\t... with lemmas %s\n\t... raw intersection: %s\n" % (synset, lemmas, intersect))

                        if len(intersect) > 0:
                            if self.debug > 0:
                                print("Found real anchor: %s (%s) with %s" % (word, intersect, synset))

                            curr_ann_attributes["anchor"] = word
                            curr_ann_attributes["lemma"] = list(intersect)[0]
                            stop = True
                            alignments += 1

            # only return correctly aligned annotations
            if curr_ann_attributes["anchor"] == "<UNDEFINED>" or curr_ann_attributes["lemma"] == "<UNDEFINED>":
                continue
            else:
                new_ann_attributes.append(curr_ann_attributes)
                new_annotations.append(curr_annotation)

        if self.debug > 0:
            print("\n--- after ---")
            for i in range(len(new_annotations)):
                print(i, new_ann_attributes[i], new_annotations[i])

        return new_ann_attributes, new_annotations, alignments


#if __name__ == "__main__":
#    aligner = SenseAligner(bn_to_wn_mapping="../resources/bn2wn_mapping.txt", debug=1)
#    print(aligner.process("Madam President , I would just like to clarify something .",
#                          [{"test": "aaa", "anchor": "Madam", "lemma": "madam"}, {"test": "bbb", "anchor": "just", "lemma": "just"}, {"test": "ccc", "anchor": ".", "lemma": "."}],
#                          ["bn:00017517n", "bn:00084526v", "bn:00052638n"]))
