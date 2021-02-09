import numpy as np
from scipy import sparse
from pathlib import Path
import os
import json
import nltk

class featureGenerator:

    def __init__(self, route="./"):
        self.cdir = Path(os.path.join(route, "features"))
        self.cdir.mkdir(exist_ok=True)

    def getNgramVocab(self, train_file,
                      ngram: "number of grams" = 1,
                      threshold: "minimum frequency require to be count" = 3) -> None:

        # get ngrams vocaburary from train_file and save results in json

        vmap = {1: "unigram", 2: "bigram", 3: "trigram"}

        if (ngram > 3):
            raise NameError("Only uni/bi/tri gram supported.")

        output_vocab = Path(os.path.join(self.cdir, f"{vmap[ngram]}_vocab.json"))

        documents = []
        with open(train_file, "r") as f:
            documents = [line[1:].strip().lower() for line in f]

        tk = nltk.WordPunctTokenizer()
        tokenize_documents = [
            list(nltk.ngrams(tk.tokenize(d), ngram)) for d in documents]

        bagOfWord = {}

        for text in tokenize_documents:
            for token in text:
                bagOfWord[token] = bagOfWord.get(token, 0) + 1

        vocab = {
            '<pad>': 0,
            '<unk>': 1
        }
        index = 2

        for token in bagOfWord:
            if (bagOfWord[token] >= threshold):
                vocab[repr(token)] = index
                index += 1

        print(f"{output_vocab} size: {len(vocab)}")

        with open(output_vocab, mode="w") as f:
            json.dump(vocab, f)

    def binaryFeature(self, feature_file,
                      ngram: "number of grams" = 1) -> None:

        # deliver binary features given ngram vocab

        types = feature_file.split(".")[-1]

        vmap = {1: "unigram", 2: "bigram", 3: "trigram"}

        if (ngram > 3):
            raise NameError("Only uni/bi/tri gram supported.")

        output_vocab = Path(os.path.join(self.cdir, f"{vmap[ngram]}_vocab.json"))
        prefix = types + f"_{vmap[ngram]}"

        if (Path.exists(output_vocab) == False):
            raise NameError(f"{output_vocab} doesn't exists")

        vocab = json.load(open(output_vocab))

        output_feature_path = Path(os.path.join(self.cdir, f"{prefix}_binary_feature.npz"))
        output_label_path = Path(os.path.join(self.cdir, f"{types}_label.npz"))

        documents = []
        labels = []
        with open(feature_file, "r") as f:
            for line in f:
                line = line.strip().lower()
                labels.append(float(line[0]))
                documents.append(line[1:])

        labels = np.asarray(labels, dtype=float)
        tk = nltk.WordPunctTokenizer()
        tokenize_documents = [list(nltk.ngrams(tk.tokenize(d), ngram))
                              for d in documents]

        M = np.zeros((len(tokenize_documents), len(vocab)))
        for index, document in enumerate(tokenize_documents):
            for token in document:
                token = repr(token)
                if (token in vocab):
                    M[index, vocab[token]] = 1
                else:
                    M[index, vocab["<unk>"]] = 1

        sparseM = sparse.csr_matrix(M)
        sparse.save_npz(output_feature_path, sparseM)
        if (Path.exists(output_label_path) == False):
            np.savez(output_label_path, labels)

        print(f"labels saved in {output_label_path}\nfeatures saved in {output_feature_path}\nfeatures space {M.shape}\n\n")

    def tfidfFeature(self, feature_file,
                     ngram: "number of grams" = 1) -> None:

        # deliver tf-idf features given ngram vocab

        types = feature_file.split(".")[-1]

        vmap = {1: "unigram", 2: "bigram", 3: "trigram"}

        if (ngram > 3):
            raise NameError("Only uni/bi/tri gram supported.")

        output_vocab = Path(os.path.join(self.cdir, f"{vmap[ngram]}_vocab.json"))
        prefix = types + f"_{vmap[ngram]}"

        output_feature_path = Path(os.path.join(self.cdir, f"{prefix}_tfidf_feature.npz"))
        output_label_path = Path(os.path.join(self.cdir, f"{types}_label.npz"))

        vocab = json.load(open(output_vocab))

        documents = []
        labels = []
        with open(feature_file, "r") as f:
            for line in f:
                line = line.strip().lower()
                labels.append(float(line[0]))
                documents.append(line[1:])

        labels = np.asarray(labels, dtype=float)
        tk = nltk.WordPunctTokenizer()
        tokenize_documents = [list(nltk.ngrams(tk.tokenize(d), ngram))
                              for d in documents]

        tf = np.zeros((len(tokenize_documents), len(vocab)))

        for i, doc in enumerate(tokenize_documents):
            for token in doc:
                token = repr(token)
                if (token in vocab):
                    tf[i, vocab[token]] += 1
                else:
                    tf[i, vocab["<unk>"]] += 1
            if (len(doc) > 0):
                tf[i, :] /= len(doc)

        idf = np.sum((tf > 0).astype(float), axis=0)
        idf = np.log((1 + len(documents)) / (1. + idf)) + 1.

        sparseM = sparse.csr_matrix(tf * idf)
        sparse.save_npz(output_feature_path, sparseM)
        if (Path.exists(output_label_path) == False):
            np.savez(output_label_path, labels)

        print(f"labels saved in {output_label_path}\nfeatures saved in {output_feature_path}\nfeatures space {tf.shape}\n\n")
