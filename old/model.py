import logging  # Setting up the loggings to monitor gensim
import multiprocessing
import os

import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from recommender import Recommender

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


class Word2VecModel(Recommender):
    def __init__(self):
        super().__init__(algorithm="word2vec")

    def train_model(
        self,
        epochs=5,
        embedding_size=128,
        window_size=150,
        min_count=10,
        number_of_negative_samples=7,
        sample=0.001,
        ns_exponent=0,
        save=False,
    ):

        model = Word2Vec(
            self.data_loader,
            sg=1,
            size=embedding_size,
            window=window_size,
            min_count=min_count,
            compute_loss=True,
            workers=multiprocessing.cpu_count() - 1,
            hs=0,
            sample=sample,
            negative=number_of_negative_samples,
            ns_exponent=ns_exponent,
            iter=epochs,
        )

        # getting the training loss value
        training_loss = model.get_latest_training_loss()
        print(f"Latest training loss: {training_loss}")

        if save:
            model.save(f"models/{self.data_loader.algorithm}/embeddings.model")
            print("Model Saved")

        model.init_sims(replace=True)

        self.embeddings = model.wv

        return model

    def predict_labels(self, classifier, x, y, test_size=0.5):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

        classifier.fit(x_train, y_train)
        y_predictions = classifier.predict(x_test)

        accuracy = accuracy_score(y_predictions, y_test)
        precision = precision_score(y_predictions, y_test, average="weighted")
        recall = recall_score(y_predictions, y_test, average="weighted")
        f1 = f1_score(y_predictions, y_test, average="weighted")

        return accuracy, precision, recall, f1

    def evaluate_embeddings(self, classifier):
        """
        Evaluate embeddings using a category and aisle classifier
        KNeighbors Classifier:
        KNeighborsClassifier(n_neighbors=k,
                             n_jobs=-1)

        Logistic Regression One-vs-Rest Classifier
        logistic_regression = LogisticRegressionCV(Cs=10,
                                           fit_intercept=True,
                                           cv=5,
                                           penalty='l2',
                                           solver='liblinear',
                                           n_jobs=-1,
                                           multi_class='ovr')
        logistic_regression = LogisticRegression(penalty='l2',
                                                C=1.0,
                                                solver='liblinear',
                                                multi_class='ovr',
                                                n_jobs=-1)
        """
        x_vector_list = []
        y_category_list = []
        y_aisle_list = []
        for key in self.embeddings.vocab.keys():
            x_vector_list.append(self.embeddings.get_vector(key))
            y_category_list.append(
                self.data_loader.product_key_to_meta(key).split("\t")[1]
            )
            y_aisle_list.append(
                self.data_loader.product_key_to_meta(key).split("\t")[2]
            )

        cat_acc_prec_rec = self.predict_labels(
            classifier, x_vector_list, y_category_list
        )
        aisle_acc_prec_rec = self.predict_labels(
            classifier, x_vector_list, y_aisle_list
        )

        print(
            f"Accuracy: {cat_acc_prec_rec[0]}, Precision: {cat_acc_prec_rec[1]}, Recall: {cat_acc_prec_rec[2]}, F1: {cat_acc_prec_rec[3]}"
        )
        print(
            f"Accuracy: {aisle_acc_prec_rec[0]}, Precision: {aisle_acc_prec_rec[1]}, Recall: {aisle_acc_prec_rec[2]}, F1: {aisle_acc_prec_rec[3]}"
        )

        return cat_acc_prec_rec, aisle_acc_prec_rec

    def create_embedding_files_for_visualization(self):
        """ create embedding files for visualization"""
        vocab_size = len(self.embeddings.vocab)

        # File Names
        target_vectors_filepath = (
            f"visualization/{self.data_loader.algorithm}_target_vectors.tsv"
        )
        target_metadata_filepath = (
            f"visualization/{self.data_loader.algorithm}_target_metadata.tsv"
        )

        # Delete if exists
        if os.path.exists(target_vectors_filepath):
            os.remove(target_vectors_filepath)

        if os.path.exists(target_metadata_filepath):
            os.remove(target_metadata_filepath)

        out_v = open(target_vectors_filepath, "w", encoding="utf-8")
        out_m = open(target_metadata_filepath, "w", encoding="utf-8")

        # Meta File Header
        out_m.write("ProductName\tCategory\tAisle" + "\n")

        for key_index in range(vocab_size):

            key = list(self.embeddings.vocab.keys())[key_index]

            # Embedding Vector
            embedding_vector = self.embeddings.get_vector(key)

            # META Input
            out_m.write(self.data_loader.product_key_to_meta(key) + "\n")
            out_v.write("\t".join([str(x) for x in embedding_vector]) + "\n")

        out_v.close()
        out_m.close()

    @staticmethod
    def _l2_norm(m, replace=False):
        """
        Return an L2-normalized version of a matrix.
        """
        dist = np.sqrt((m ** 2).sum(-1))[..., np.newaxis]
        if replace:
            m /= dist
            return m
        else:
            return (m / dist).astype(np.float32)

    def generate_candidates(self, user_id, given_items):
        product_ID_Vectors = [
            _l2_norm(model.wv.get_vector(product_ID), replace=False)
            for product_ID in product_IDs
        ]

        return None

    def rank_candidates(self, k, candidate_list):

        return None
