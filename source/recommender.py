import math
import time

import numpy as np
from gensim.matutils import argsort


class Recommender(object):
    def __init__(
        self,
        algorithm,
        item_key_mapping,
        user_item_frequency,
        number_of_candidates=-1,
        predict_k=-1,
    ):
        self.algorithm = algorithm
        self.n_items = len(item_key_mapping)
        self.item_key_mapping = item_key_mapping
        self.reverse_item_key_mapping = {
            index: item_key for item_key, index in item_key_mapping.items()
        }
        self.user_item_frequency = user_item_frequency

        # if not specified, return all items as candidates and predict all (during evaluation we want to work with all items to calculate AUC and NDCG)
        self.number_of_candidates = (
            self.n_items if number_of_candidates == -1 else number_of_candidates
        )
        self.predict_k = self.n_items if predict_k == -1 else predict_k

    def generate_candidates(self, given_items):

        return None

    def rank_candidates(self, user_id, candidate_list):

        return None

    def predict_items(self, user_id, given_items):
        candidate_scores = self.generate_candidates(given_items=given_items)
        ranked_candidates = self.rank_candidates(
            user_id=user_id, candidate_scores=candidate_scores
        )

        return ranked_candidates

    def evaluate_transaction(self, user_id, given_items, test_items, k=20):
        # get the predicted items and their scores
        item_scores = self.predict_items(user_id=user_id, given_items=given_items)
        # create a list of item scores
        # predicted_item_scores = np.array(list(item_scores.values()))
        # create a mapping for items to index in the scores list
        # item_indices = {key: index for index, key in enumerate(item_scores.keys())}

        # separate the target items from the other items
        negative_index = np.ones(self.n_items)
        mask_items = [self.item_key_mapping[item] for item in test_items]
        negative_index[mask_items] = 0
        target_item_scores = item_scores[mask_items]
        negative_items = item_scores[negative_index > 0]

        # calculate the auc and ndcg
        n_negative = len(negative_items)
        false_predictions = (
            target_item_scores.reshape(1, len(target_item_scores))
            <= negative_items.reshape(n_negative, 1)
        ).sum(axis=0)
        auc = (n_negative - false_predictions) / n_negative
        ndcg = 1.0 / np.log2(2 + false_predictions)

        # Precision and Recall @ K
        top_k_items = argsort(item_scores, topn=k, reverse=True)
        recall_at_k = self.recall(mask_items, top_k_items, k=k)
        precision_at_k = self.precision(mask_items, top_k_items, k=k)

        return auc, ndcg, recall_at_k, precision_at_k

    def evaluate(self, test_transactions, k=20, within_basket=True, verbose=True):
        start_time = time.time()

        # more efficient than interating over numpy array
        test_transactions = list(test_transactions)
        metrics = []
        min_transaction_items = 2
        # after removing transactions with less than MIN_TRANSACTION_ITEMS
        actual_transaction_length = len(test_transactions)
        if verbose:
            print(f"{actual_transaction_length} transactions to evaluate.")

        for test_transaction in test_transactions:
            # user id is always first in list, then all the purchased items
            user_id = test_transaction[0]
            items = [
                item for item in test_transaction[1:] if item in self.item_key_mapping
            ]

            if within_basket:
                if len(items) < min_transaction_items:
                    actual_transaction_length -= 1
                    continue

                half = math.ceil(len(items) / 2)
                given_item_ids = items[:half]  # given items
                hold_out_item_ids = items[half:]  # test items
            else:
                if (
                    user_id not in self.user_item_frequency
                    or len(items) < min_transaction_items
                ):
                    actual_transaction_length -= 1
                    continue
                given_item_ids = list(self.user_item_frequency[user_id].keys())
                given_item_ids = [
                    item for item in given_item_ids if item in self.item_key_mapping
                ]
                hold_out_item_ids = items

            _auc, _ndcg, _recall_at_k, _precision_at_k = self.evaluate_transaction(
                user_id=user_id,
                given_items=given_item_ids,
                test_items=hold_out_item_ids,
                k=k,
            )

            metrics.append([_auc.mean(), _ndcg.mean(), _recall_at_k, _precision_at_k])

        actual_transaction_length = len(metrics)
        if verbose:
            print(f"Evaluated {actual_transaction_length} transactions.")

        metrics = np.array(metrics).mean(axis=0)
        auc = round(metrics[0], 4)
        ndcg = round(metrics[1], 4)
        recall_at_k = round(metrics[2], 4)
        precision_at_k = round(metrics[3], 4)

        if verbose:
            print(f"Took {round((time.time()-start_time)/60., 3)} minutes.")

            print(f"AUC:              {auc}")
            print(f"NDCG:             {ndcg}")
            print(f"Recall at {k}:     {recall_at_k}")
            print(f"Precision at {k}:  {precision_at_k}")

        return auc, ndcg, recall_at_k, precision_at_k

    @staticmethod
    def precision(actual, predicted, k):
        actual_set = set(actual)
        predicted_set = set(predicted[:k])
        result = len(actual_set & predicted_set) / float(k)
        return result

    @staticmethod
    def recall(actual, predicted, k):
        actual_set = set(actual)
        predicted_set = set(predicted[:k])
        result = len(actual_set & predicted_set) / float(len(actual_set))
        return result
