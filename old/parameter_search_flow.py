import math
import multiprocessing
import os
import statistics
import time

import numpy as np
import pandas as pd
from gensim.matutils import argsort
from gensim.models.word2vec import FAST_VERSION, Word2Vec
from metaflow import FlowSpec, Parameter, batch, conda, conda_base, retry, step
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from item2vec_recommender import Item2VecRecommender
from iterator import TransactionDataIterator
from utils import (
    generate_user_item_interactions,
    get_product_key_conversion,
    load_data,
    precision,
    product_key_to_meta,
    product_key_to_name,
    recall,
)


def train_model(
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

    cores = multiprocessing.cpu_count()
    # print(f"Fast Version: {FAST_VERSION}")
    # print(f"Running on {cores} cores.")

    model = Word2Vec(
        train_data,
        sg=1,
        size=embedding_size,
        window=window_size,
        min_count=min_count,
        compute_loss=True,
        workers=cores,
        hs=0,
        sample=sample,
        negative=number_of_negative_samples,
        ns_exponent=ns_exponent,
        iter=epochs,
    )

    model.init_sims(replace=True)

    return model


def predict_labels(classifier, x, y, test_size=0.5):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=0
    )

    classifier.fit(x_train, y_train)
    y_predictions = classifier.predict(x_test)

    f1_micro = round(f1_score(y_predictions, y_test, average="micro"), 4)
    f1_macro = round(f1_score(y_predictions, y_test, average="macro"), 4)
    f1_weighted = round(f1_score(y_predictions, y_test, average="weighted"), 4)

    return f1_micro, f1_macro, f1_weighted


def evaluate_embeddings(
    embedding, mapping, product_key_conversion, k_neighbors=10, verbose=True
):
    y_category_list = []
    y_aisle_list = []

    for key in mapping.keys():
        y_category_list.append(
            product_key_to_meta(product_key_conversion, key).split("\t")[1]
        )
        y_aisle_list.append(
            product_key_to_meta(product_key_conversion, key).split("\t")[2]
        )

    # K Neighbors Classifier
    k_neighbors_classifier = KNeighborsClassifier(n_neighbors=k_neighbors, n_jobs=-1)

    category_f1 = predict_labels(
        classifier=k_neighbors_classifier, x=embedding, y=y_category_list
    )
    aisle_f1 = predict_labels(
        classifier=k_neighbors_classifier, x=embedding, y=y_aisle_list
    )

    if verbose:
        print(
            f"Category - Micro: {category_f1[0]}, Macro: {category_f1[1]}, Weighted: {category_f1[2]}"
        )
        print(
            f"Aisle    - Micro: {aisle_f1[0]}, Macro: {aisle_f1[1]}, Weighted: {aisle_f1[2]}"
        )

    return category_f1, aisle_f1


def parameter_search(
    train_data,
    validation_data,
    user_item_frequency,
    params,
    product_key_conversion,
    k_neighbors=10,
    predict_k=10,
):
    """ Parameter Search """
    print(
        params["epochs"],
        params["window_sizes"],
        params["samples"],
        params["ns_exponents"],
        params["embedding_size"],
        params["numbers_of_negative_samples"],
    )

    start_time = time.time()
    print(f"Parameters: {params}")

    # Train the model
    model = train_model(
        train_data,
        epochs=params["epochs"],
        embedding_size=params["embedding_size"],
        window_size=params["window_sizes"],
        min_count=10,
        number_of_negative_samples=params["numbers_of_negative_samples"],
        sample=params["samples"],
        ns_exponent=params["ns_exponents"],
        save=False,
    )

    # Create a matrix filled with embeddings of all items considered.
    mapping = {item_key: index for index, item_key in enumerate(model.wv.index2word)}
    mapping_back = {index: item_key for item_key, index in mapping.items()}
    embedding = [model.wv[key] for key in mapping.keys()]

    # Evaluate Embeddings
    category_f1, aisle_f1 = evaluate_embeddings(
        embedding=embedding,
        mapping=mapping,
        product_key_conversion=product_key_conversion,
        k_neighbors=k_neighbors,
        verbose=False,
    )

    # Recommender
    item2vec_recommender = Item2VecRecommender(
        algorithm="item2vec",
        item_key_mapping=mapping,
        user_item_frequency=user_item_frequency,
        embedding_vectors=model.wv.vectors,
        context_vectors=model.trainables.syn1neg,
    )

    within_basket_validation = item2vec_recommender.evaluate(
        validation_data, k=predict_k, within_basket=True, verbose=False
    )

    next_basket_validation = item2vec_recommender.evaluate(
        validation_data, k=predict_k, within_basket=False, verbose=False
    )

    results = (
        params["epochs"],
        params["window_sizes"],
        params["samples"],
        params["ns_exponents"],
        params["embedding_size"],
        params["numbers_of_negative_samples"],
        category_f1[0],
        aisle_f1[0],
        category_f1[1],
        aisle_f1[1],
        within_basket_validation[0],
        within_basket_validation[1],
        within_basket_validation[2],
        within_basket_validation[3],
        next_basket_validation[0],
        next_basket_validation[1],
        next_basket_validation[2],
        next_basket_validation[3],
    )

    end = time.time()
    print(f"Took {round((time.time()-start_time)/60., 3)} minutes.")

    return results


@conda_base(
    libraries={
        "gensim": "3.8.0",
        "numpy": "1.18.5",
        "pandas": "1.0.4",
        "scipy": "1.4.1",
        "scikit-learn": "0.23.1",
    },
    python="3.7",
)
class ParameterSearchFlow(FlowSpec):
    """
    This Flow does Parameter Search
    """

    small = Parameter("small", help="Recommender to Train", default=False)
    algorithm = Parameter("algorithm", help="Recommender to Train", default="item2vec")

    @step
    def start(self):
        """ Start """
        self.data_loader = load_data(small_data=self.small)
        self.train_data = self.data_loader["train"]
        self.validation_data = self.data_loader["validation"]
        # self.test_data = self.data_loader["test"]
        self.item_metadata = self.data_loader["metadata"]

        self.product_key_conversion = get_product_key_conversion(
            metadata=self.item_metadata, save=False
        )

        self.train_data_iter = TransactionDataIterator(data=self.train_data)
        self.user_item_frequency = generate_user_item_interactions(
            train_data=self.train_data
        )

        param_grid_values = {
            "epochs": [5, 50],
            "window_sizes": [100],
            "samples": [0.001, 0.01, 0.1],
            "ns_exponents": [0.25, 0.5, 0.75],
            "embedding_size": [128],
            "numbers_of_negative_samples": [7],
        }
        # param_grid_values = {'epochs': [1],
        #                     'window_sizes': [100],
        #                     'samples': [0.01],
        #                     'ns_exponents': [0.5],
        #                     'embedding_size': [64],
        #                     'numbers_of_negative_samples': [7]}

        self.parameter_grid = ParameterGrid(param_grid_values)

        self.next(self.parameter_search, foreach="parameter_grid")

    @retry
    @batch(memory=32000, cpu=16)
    @step
    def parameter_search(self):
        """ Parameter Search """
        parameters = self.input

        self.result = parameter_search(
            train_data=self.train_data_iter,
            validation_data=self.validation_data,
            params=parameters,
            user_item_frequency=self.user_item_frequency,
            product_key_conversion=self.product_key_conversion,
            k_neighbors=10,
            predict_k=10,
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        """ Join """
        self.results = [input.result for input in inputs]

        # Sorting by best within basket AUC
        self.results.sort(key=lambda x: x[10], reverse=True)

        with open(f"{self.algorithm}_parameter_search.txt", "w") as f:
            for line in self.results:
                f.write(f"{str(line)}\n")

        results_df = pd.DataFrame(
            self.results,
            columns=[
                "Epoch",
                "Window Size",
                "Sample",
                "NS Exponent",
                "Embedding Size",
                "Number of Negative Samples",
                "F1 Macro Category",
                "F1 Macro Aisle",
                "F1 Micro Category",
                "F1 Micro Aisle",
                "Within-basket AUC",
                "Within-basket NDCG",
                "Within-basket Recall",
                "Within-basket Precision",
                "Next-basket AUC",
                "Next-basket NDCG",
                "Next-basket Recall",
                "Next-basket Precision",
            ],
        )
        results_df.to_csv(f"results/{self.algorithm}/results.csv", index=False)

        self.next(self.end)

    @step
    def end(self):
        """ End """
        pass


if __name__ == "__main__":
    ParameterSearchFlow()
