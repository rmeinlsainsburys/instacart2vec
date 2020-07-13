import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import ParameterGrid

from data_loader import DataLoader
from item2vec_embeddings import Item2VecEmbeddings
from item2vec_recommender import Item2VecRecommender


def save_results(results, algorithm):
    results_path = f"results/{algorithm}_results.csv"

    results_df = pd.DataFrame(
        results,
        columns=[
            "Epoch",
            "Window Size",
            "Sample",
            "NS Exponent",
            "Embedding Size",
            "Number of Negative Samples",
            "F1 Micro Category",
            "F1 Micro Aisle",
            "F1 Macro Category",
            "F1 Macro Aisle",
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
    results_df.to_csv(results_path, index=False)
    print(f"Saved to {results_path}")


def mean_confidence_interval(data, confidence=0.95):
    """ Standard t-test over mean."""
    array = 1.0 * np.array(data)
    array_length = len(array)
    if array_length == 1:
        mean = array[0]
        h = math.nan
    else:
        mean, std_error_mean = np.mean(array), sp.stats.sem(array)
        h = std_error_mean * sp.stats.t._ppf((1 + confidence) / 2.0, array_length - 1)

    return mean, mean - h, mean + h, h


def parameter_search(
    embeddings,
    train_data_iterator,
    validation_data,
    user_item_frequency,
    parameter_grid,
    confidence_iterations=5,
    min_count=10,
    k_neighbors=10,
    predict_k=10,
):
    results = []
    parameter_search_path = f"results/{algorithm}_parameter_search.txt"
    if os.path.exists(parameter_search_path):
        with open(parameter_search_path) as file:
            results = file.readlines()
        results = [eval(p.strip()) for p in results]
        print(f"Previous results loaded: {results}")

    for params in parameter_grid:
        params_tuple = (
            params["epochs"],
            params["window_sizes"],
            params["samples"],
            params["ns_exponents"],
            params["embedding_size"],
            params["numbers_of_negative_samples"],
        )
        # TODO: uncomment
        # if any([set(params_tuple).issubset(parameter) for parameter in results]):
        #     print(f"{params_tuple} already exists. Skipping.")
        #     continue

        confidence_results = []

        for confidence_iteration in range(confidence_iterations):
            print(
                f"Starting iteration {confidence_iteration+1}/{confidence_iterations} of evaluation for: {params_tuple}"
            )

            start_time = time.time()
            print(f"Parameters: {params}")

            # Train the model
            embeddings.train_model(
                train_data=train_data_iterator,
                epochs=params["epochs"],
                embedding_size=params["embedding_size"],
                window_size=params["window_sizes"],
                min_count=min_count,
                number_of_negative_samples=params["numbers_of_negative_samples"],
                sample=params["samples"],
                ns_exponent=params["ns_exponents"],
                save=True,
            )

            # Evaluate Embeddings
            category_f1, aisle_f1 = embeddings.evaluate_embeddings(
                k_neighbors=k_neighbors
            )

            # Create the Recommender
            item2vec_recommender = Item2VecRecommender(
                algorithm=embeddings.algorithm,
                item_key_mapping=embeddings.mapping,
                user_item_frequency=user_item_frequency,
                embedding_vectors=embeddings.embedding_vectors,
                context_vectors=embeddings.context_vectors,
                user_vectors=embeddings.user_vectors,
            )

            # Within Basket Recommendations
            within_basket_validation = item2vec_recommender.evaluate(
                validation_data, k=predict_k, within_basket=True
            )

            # Next Basket Recommendations
            next_basket_validation = item2vec_recommender.evaluate(
                validation_data, k=predict_k, within_basket=False
            )

            confidence_results.append(
                (
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
            )

        unzipped_confidence_results = [
            list(confidence_result) for confidence_result in zip(*confidence_results)
        ]
        category_f1_micro_mean, _, _, category_f1_micro_h = mean_confidence_interval(
            unzipped_confidence_results[0], confidence=0.95
        )
        aisle_f1_micro_mean, _, _, aisle_f1_micro_h = mean_confidence_interval(
            unzipped_confidence_results[1], confidence=0.95
        )
        category_f1_macro_mean, _, _, category_f1_macro_h = mean_confidence_interval(
            unzipped_confidence_results[2], confidence=0.95
        )
        aisle_f1_macro_mean, _, _, aisle_f1_macro_h = mean_confidence_interval(
            unzipped_confidence_results[3], confidence=0.95
        )
        within_basket_auc_mean, _, _, within_basket_auc_h = mean_confidence_interval(
            unzipped_confidence_results[4], confidence=0.95
        )
        within_basket_ndcg_mean, _, _, within_basket_ndcg_h = mean_confidence_interval(
            unzipped_confidence_results[5], confidence=0.95
        )
        (
            within_basket_recall_mean,
            _,
            _,
            within_basket_recall_h,
        ) = mean_confidence_interval(unzipped_confidence_results[6], confidence=0.95)
        (
            within_basket_precision_mean,
            _,
            _,
            within_basket_precision_h,
        ) = mean_confidence_interval(unzipped_confidence_results[7], confidence=0.95)
        next_basket_auc_mean, _, _, next_basket_auc_h = mean_confidence_interval(
            unzipped_confidence_results[8], confidence=0.95
        )
        next_basket_ndcg_mean, _, _, next_basket_ndcg_h = mean_confidence_interval(
            unzipped_confidence_results[9], confidence=0.95
        )
        next_basket_recall_mean, _, _, next_basket_recall_h = mean_confidence_interval(
            unzipped_confidence_results[10], confidence=0.95
        )
        (
            next_basket_precision_mean,
            _,
            _,
            next_basket_precision_h,
        ) = mean_confidence_interval(unzipped_confidence_results[11], confidence=0.95)

        results.append(
            (
                params["epochs"],
                params["window_sizes"],
                params["samples"],
                params["ns_exponents"],
                params["embedding_size"],
                params["numbers_of_negative_samples"],
                f"{round(category_f1_micro_mean, 4)} +/- {round(category_f1_micro_h, 4)}",
                f"{round(aisle_f1_micro_mean, 4)} +/- {round(aisle_f1_micro_h, 4)}",
                f"{round(category_f1_macro_mean, 4)} +/- {round(category_f1_macro_h, 4)}",
                f"{round(aisle_f1_macro_mean, 4)} +/- {round(aisle_f1_macro_h, 4)}",
                f"{round(within_basket_auc_mean, 4)} +/- {round(within_basket_auc_h, 4)}",
                f"{round(within_basket_ndcg_mean, 4)} +/- {round(within_basket_ndcg_h, 4)}",
                f"{round(within_basket_recall_mean, 4)} +/- {round(within_basket_recall_h, 4)}",
                f"{round(within_basket_precision_mean, 4)} +/- {round(within_basket_precision_h, 4)}",
                f"{round(next_basket_auc_mean, 4)} +/- {round(next_basket_auc_h, 4)}",
                f"{round(next_basket_ndcg_mean, 4)} +/- {round(next_basket_ndcg_h, 4)}",
                f"{round(next_basket_recall_mean, 4)} +/- {round(next_basket_recall_h, 4)}",
                f"{round(next_basket_precision_mean, 4)} +/- {round(next_basket_precision_h, 4)}",
            )
        )

        with open(parameter_search_path, "w") as f:
            for line in results:
                f.write(f"{str(line)}\n")

        print(f"Took {round((time.time()-start_time)/60., 3)} minutes\n")

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--parameter_grid_location",
        dest="parameter_grid_location",
        required=True,
        type=str,
    )
    parser.add_argument("-s", "--small", dest="small", action="store_true")
    parser.set_defaults(small=False)
    parser.add_argument("--with_meta", dest="with_meta", action="store_true")
    parser.add_argument("--with_user", dest="with_user", action="store_true")
    parser.set_defaults(with_meta=False)
    parser.set_defaults(with_user=False)
    parser.add_argument(
        "--use_file_iterator", dest="use_file_iterator", action="store_true"
    )
    parser.set_defaults(use_file_iterator=False)
    parser.add_argument(
        "-it",
        "--confidence_iterations",
        dest="confidence_iterations",
        default=5,
        type=int,
    )
    parser.add_argument("--min_count", dest="min_count", default=10, type=int)
    parser.add_argument("--k_neighbors", dest="k_neighbors", default=10, type=int)
    parser.add_argument("--predict_k", dest="predict_k", default=10, type=int)
    parser.add_argument("--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    parser.add_argument("--use_test_set", dest="use_test_set", action="store_true")
    parser.set_defaults(use_test_set=False)
    args = parser.parse_args()
    print(args)

    algorithm = "item2vec"
    if args.with_meta:
        algorithm = "meta-" + algorithm
    if args.with_user:
        algorithm = "user-" + algorithm

    print(f"Using {algorithm} algorithm.")

    data_loader = DataLoader(
        algorithm=algorithm,
        small_data=args.small,
        with_meta=args.with_meta,
        with_user=args.with_user,
        use_file_iterator=args.use_file_iterator,
    )

    with open(args.parameter_grid_location) as file:
        parameter_grid_values = json.load(file)

    parameter_grid = ParameterGrid(parameter_grid_values)

    embeddings = Item2VecEmbeddings(
        algorithm=algorithm,
        product_key_conversion=data_loader.product_key_conversion,
        with_meta=args.with_meta,
        with_user=args.with_user,
    )

    results = parameter_search(
        embeddings=embeddings,
        train_data_iterator=data_loader.train_data_iterator,
        validation_data=data_loader.validation_data
        if not args.use_test_set
        else data_loader.test_data,
        user_item_frequency=data_loader.user_item_frequency,
        parameter_grid=parameter_grid,
        confidence_iterations=args.confidence_iterations,
        min_count=args.min_count,
        k_neighbors=args.k_neighbors,
        predict_k=args.predict_k,
    )

    if args.save:
        save_results(results=results, algorithm=algorithm)
