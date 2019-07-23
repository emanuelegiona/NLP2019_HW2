# NLP 2018/2019 ([more][1])

## Homework 2

Sense embedding is a technique for language modeling and feature learning applied in order to obtain dense vectors for words senses, instead of relying on large sparse vectors as the ones produced via a one-hot enconding approach.

The need for vectors representing the different senses of a word is driven by the high degree of polysemy that is proper of natural languages, which a single vector per word fails to address when such word is ambiguous.

Similarly to word vectors, sense embeddings can be built through neural approaches equivalent to Word2Vec, and an example of this technique is the one presented in **SensEmbed: Learning Sense Embeddings for Word and Relational Similarity**, which is the reference paper for this homework assignment. The authors implement a Word2Vec model in the **Continuous Bag of Words** (CBOW) variant, for which a held-out word has to be predicted based on a context window. The aforementioned model is then trained on a corpus which is automatically annotated with BabelNet synsets representing word senses, being able to also exploit a large structured knowledge source additionally to raw text corpora.

[Continue reading][2]

[1]: http://naviglinlp.blogspot.com/
[2]: ./hw2_report_anonymous.pdf
