from collections import namedtuple
import tarfile
from lxml import etree

import utils as u
import sensealigner


class XmlParser:
    """
    Class to parse XML files compliant to EuroSense's format.
    """

    def __init__(self, file, compressed=True):
        self.file = file
        self.compressed = compressed

    def __enter__(self):
        if self.compressed:
            self.file_handle = tarfile.open(self.file)
            self.xml = self.file_handle.extractfile("EuroSense/eurosense.v1.0.high-precision.xml")
        else:
            self.file_handle = open(self.file, mode="rb")
            self.xml = self.file_handle
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.xml
        self.file_handle.close()

    def parse(self, languages, replace=False, bn_to_wn_mapping=None):
        """
        Parses sentences of the given languages.

        In case replace=False: for each sentence, returns its attributes, text, text attributes, annotations, annotations attributes.
        In case replace=True: for each sentence, returns the processed sentence according the replacing rule.

        :param languages: languages to be considered (in ISO code), as List
        :param replace: True: replaces anchors with lemma_annotation; False: nothing (default)
        :param bn_to_wn_mapping: dictionary mapping BabelNet IDs to WordNet IDs
        :return: appropriate return, in a generator fashion
        """
        assert (replace is False) or (replace and bn_to_wn_mapping is not None), "BabelNet to WordNet mapping must be provided if replace=True"

        for event, sentence in etree.iterparse(self.xml, tag="sentence"):
            proc_sentences = []
            sentence_attrs = sentence.attrib
            text_attrs, text, annotations, annotations_attrs = [], [], [], []

            if event == "end":
                for element in sentence:
                    if element.tag == "text" and element.get("lang") in languages:
                        text_attrs.append(element.attrib)
                        text.append(element.text)

                    if element.tag == "annotations":
                        for annotation in element:
                            if annotation.get("lang") in languages:
                                annotations_attrs.append(annotation.attrib)
                                annotations.append(annotation.text)

                if not replace:
                    yield sentence_attrs, text, text_attrs, annotations, annotations_attrs

                else:
                    # used to keep track of possible replacements in order to select the longest mention
                    Replacement = namedtuple("Replacement", "anchor synset lemma")

                    # iterate over all texts, one for each language selected
                    for single_text, single_attrs in zip(text, text_attrs):

                        # skip processing of null texts (i.e. sentence id = 2031)
                        if single_text is None:
                            continue

                        proc_sentence = []
                        for word in single_text.split(" "):
                            curr_replacement = Replacement(anchor=[], synset="<NO_SYNSET>", lemma="")

                            for annotation, ann_attrs in zip(annotations, annotations_attrs):

                                # no need to parse annotations of another language than the text's
                                if ann_attrs["lang"] != single_attrs["lang"]:
                                    continue

                                curr_anchor = ann_attrs["anchor"].split(" ")
                                if word in curr_anchor:
                                    # longest mention is preferred in case of multiple annotations for the same word
                                    if len(curr_anchor) > len(curr_replacement.anchor):
                                        curr_replacement = Replacement(anchor=curr_anchor,
                                                                       synset=annotation,
                                                                       lemma=ann_attrs["lemma"])

                            # no annotation for this word
                            if curr_replacement.synset == "<NO_SYNSET>":
                                proc_sentence.append(word)

                            # annotation found and word is the last in its mention
                            elif curr_replacement.synset != "<NO_SYNSET>" and \
                                    word == curr_replacement.anchor[-1]:

                                # build the lemma_synset format for the whole mention
                                replacement_word = "%s_%s" % (curr_replacement.lemma.replace(" ", "_"),
                                                              curr_replacement.synset)
                                proc_sentence.append(replacement_word)

                        # form a string concatenated by space
                        proc_sentences.append(" ".join(proc_sentence))

                    yield proc_sentences

                sentence.clear()


class XmlWriter:
    """
    Class to write XML files compliant to EuroSense's format.
    """

    def __init__(self, file):
        self.file = file

        with open(self.file, mode="w") as xml:
            xml.write("<?xml version='1.0' encoding='UTF-8'?>\n")
            xml.write("<xml>\n")

    def __enter__(self):
        self.file_handle = open(self.file, mode="ab")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, sentence_attrs, text, text_attrs, annotations, annotations_attrs):
        """
        Writes a sentence.
        :param sentence_attrs: attributes of the sentence
        :param text: texts to be set to the sentence, as List
        :param text_attrs: attributes of texts, as List of Dicts
        :param annotations: texts of annotations, as List
        :param annotations_attrs: attributes for annotations, as List of Dicts
        :return: None
        """
        assert len(text) == len(text_attrs), "Number of texts and texts attributes must match"
        assert len(annotations) == len(annotations_attrs), "Number of annotations and annotations attributes must match"

        sentence_tag = etree.Element("sentence", sentence_attrs)
        for i in range(len(text)):
            text_tag = etree.SubElement(sentence_tag, "text", attrib=text_attrs[i])
            text_tag.text = text[i]

        annotations_tag = etree.SubElement(sentence_tag, "annotations")
        for i in range(len(annotations)):
            annotation_tag = etree.SubElement(annotations_tag, "annotation", attrib=annotations_attrs[i])
            annotation_tag.text = annotations[i]

        sentence_tree = etree.ElementTree(sentence_tag)
        sentence_tree.write(self.file_handle, encoding="utf-8", pretty_print=True)

    def close(self):
        self.file_handle.close()
        with open(self.file, 'a') as xml:
            xml.write("</xml>\n")


if __name__ == "__main__":
    bn2wn_mapping = "../resources/bn2wn_mapping.txt"
    logfile = "../logs/parsing.log"
    print_every = 50_000

    with \
            open(logfile, "w") as log,\
            XmlParser("../resources/eurosense.v1.0.high-precision.tar.gz") as parser,\
            XmlWriter("../resources/eurosense.v1.0.english_only.xml") as writer:

        aligner = sensealigner.SenseAligner(bn_to_wn_mapping=bn2wn_mapping)
        u.log_message(log, "Started parsing")

        count = 0
        tot_annotations = 0
        tot_discarded = 0
        tot_alignments = 0
        for sa, t, ta, a, at in parser.parse(languages=["en"]):

            # english-only sentences, so it's just the first text
            at_aligned, a_aligned, alignments = aligner.process(sentence=t[0],
                                                                annotations_attributes=at,
                                                                annotations=a)

            # stats
            discarded = len(a) - len(a_aligned)
            tot_discarded += discarded
            tot_annotations += len(a)
            tot_alignments += alignments

            writer.write(sentence_attrs=sa,
                         text=t,
                         text_attrs=ta,
                         annotations=a_aligned,
                         annotations_attrs=at_aligned)
            count += 1

            if count % print_every == 0:
                u.log_message(log,"Sentences parsed: %d | Alignments performed: %d (%.2f%%) | Annotations discarded: %d (%.2f%%)" % (count,
                                                                                                                                     tot_alignments,
                                                                                                                                     float(tot_alignments)/tot_annotations * 100,
                                                                                                                                     tot_discarded,
                                                                                                                                     float(tot_discarded)/tot_annotations * 100))

        u.log_message(log, "Finished parsing.")
        u.log_message(log, "Sentences parsed: %d | Alignments performed: %d (%.2f%%) | Annotations discarded: %d / %d (%.2f%%)" % (count,
                                                                                                                                   tot_alignments,
                                                                                                                                   float(tot_alignments)/tot_annotations * 100,
                                                                                                                                   tot_discarded,
                                                                                                                                   tot_annotations,
                                                                                                                                   float(tot_discarded)/tot_annotations * 100))
