import multiprocessing

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec

from embeddings import Embeddings


class Item2VecEmbeddings(Embeddings):
    def __init__(
        self, algorithm, product_key_conversion, with_meta=False, with_user=False
    ):
        super().__init__(
            algorithm=algorithm,
            product_key_conversion=product_key_conversion,
            with_meta=with_meta,
            with_user=with_user,
        )

    def get_model_vectors(self, model):
        if self.with_meta:
            to_trim = [
                (index, item_key)
                for index, item_key in enumerate(model.wv.index2word)
                if (item_key.startswith("category_") or item_key.startswith("aisle_"))
            ]
            indices_to_trim, words_to_trim = list(zip(*to_trim))
            indices_to_trim = list(indices_to_trim)
            words_to_trim = list(words_to_trim)

            print(
                f"Removing {len(words_to_trim)} categories from the model: {words_to_trim}"
            )

            self.model_metadata_vectors = {}

            for word in words_to_trim:
                self.model_metadata_vectors[word] = model.wv[word]
                del model.wv.vocab[word]

            self.embedding_vectors = np.delete(
                model.wv.vectors, indices_to_trim, axis=0
            )
            self.context_vectors = np.delete(
                model.trainables.syn1neg, indices_to_trim, axis=0
            )

            for index in sorted(indices_to_trim, reverse=True):
                del model.wv.index2word[index]

            test_index2word = set(model.wv.index2word)
            for word in words_to_trim:
                assert word not in model.wv.vocab
                assert word not in test_index2word
        else:
            self.embedding_vectors = model.wv.vectors
            self.context_vectors = model.trainables.syn1neg

        if self.with_user:
            self.user_vectors = model.docvecs
        else:
            self.user_vectors = None

        self.model = model

    @staticmethod
    def train_word2vec(
        sentences,
        iter,
        size,
        window,
        ns_exponent,
        negative,
        min_count,
        sample,
        workers,
        sg=1,
        hs=0,
    ):
        from gensim.models.word2vec import FAST_VERSION

        print(f"Fast Word2Vec Version: {FAST_VERSION}.")

        model = Word2Vec(
            sentences=sentences,
            sg=sg,
            size=size,
            window=window,
            min_count=min_count,
            workers=workers,
            hs=hs,
            sample=sample,
            negative=negative,
            ns_exponent=ns_exponent,
            iter=iter,
        )

        return model

    @staticmethod
    def train_doc2vec(
        documents,
        epochs,
        vector_size,
        window,
        ns_exponent,
        negative,
        min_count,
        sample,
        workers,
        dm=0,
        dm_mean=None,
        dm_concat=0,
        dbow_words=1,
        hs=0,
    ):
        from gensim.models.doc2vec import FAST_VERSION

        print(f"Fast Doc2Vec Version: {FAST_VERSION}")

        # PV-DBOW: dm=0, dbow_words=1
        # PV-DM modes without concatenation dm=1, dm_concat=0
        model = Doc2Vec(
            documents=documents,
            dm=dm,  # 1,
            dm_mean=dm_mean,  # if 0, it uses sum of context vectors instead of average
            dm_concat=dm_concat,
            dbow_words=dbow_words,  # if 1 it trains word vectors as well, if 0 it only trains doc vectors
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            hs=hs,
            sample=sample,
            negative=negative,
            ns_exponent=ns_exponent,
            epochs=epochs,
        )

        return model

    def train_model(
        self,
        train_data,
        epochs,
        embedding_size,
        window_size,
        ns_exponent,
        number_of_negative_samples,
        min_count,
        sample,
        save=False,
    ):
        num_workers = multiprocessing.cpu_count()

        if not self.with_user:
            model = self.train_word2vec(
                sentences=train_data,
                iter=epochs,
                size=embedding_size,
                window=window_size,
                ns_exponent=ns_exponent,
                negative=number_of_negative_samples,
                min_count=min_count,
                sample=sample,
                workers=num_workers,
            )
        else:
            model = self.train_doc2vec(
                documents=train_data,
                epochs=epochs,
                vector_size=embedding_size,
                window=window_size,
                ns_exponent=ns_exponent,
                negative=number_of_negative_samples,
                min_count=min_count,
                sample=sample,
                workers=num_workers,
            )

        model.init_sims(replace=True)

        if save:
            model.save(f"models/{self.algorithm}/embeddings.model")
            print("Model Saved")

        # Gets the embedding and context vectors, if using metadata it also filters the model so it only contains the item vectors
        self.get_model_vectors(model)

        # create a mapping from "product_id" to "index"
        self.create_mapping()

        assert (
            len(self.mapping)
            == len(self.embedding_vectors)
            == len(self.context_vectors)
        )

    def load_model(self, model_path):
        if self.with_user:
            print("Loading Doc2Vec model...")
            model = Doc2Vec.load(model_path)
        else:
            print("Loading Word2Vec model...")
            model = Word2Vec.load(model_path)

        # Gets the embedding and context vectors, if using metadata it also filters the model so it only contains the item vectors
        self.get_model_vectors(model)

        # create a mapping from "product_id" to "index"
        self.create_mapping()

        assert (
            len(self.mapping)
            == len(self.embedding_vectors)
            == len(self.context_vectors)
        )
