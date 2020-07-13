from collections import OrderedDict

import numpy as np

from recommender import Recommender


class MostPopularRecommender(Recommender):
    def __init__(self, item_key_mapping, item_frequency, user_item_frequency):
        super().__init__(
            algorithm="most_popular",
            item_key_mapping=item_key_mapping,
            user_item_frequency=user_item_frequency,
        )
        # create list out of an ordered dict, sorted by the item frequency descendingly: [(item1, 999), (item2, 998), ...]
        self.item_frequency = item_frequency  # list(OrderedDict(sorted(item_frequency.items(), key=lambda t: t[1], reverse=True)).items())

    def generate_candidates(self, given_items):
        np.random.seed(0)

        candidate_list = np.random.rand(self.n_items) * 1e-2
        candidate_list += self.item_frequency.copy()

        return self.item_frequency

    def rank_candidates(self, user_id, candidate_scores):
        # if self.predict_k < self.n_items do candidate_list[:predict_k]; otherwise its equal to self.n_items and I can apply it to all
        # ranked_candidates = {product_id: frequency for (product_id, frequency) in candidate_list[:self.predict_k]}

        return candidate_scores


class MostPopularForUserRecommender(Recommender):
    def __init__(self, item_key_mapping, user_item_frequency):
        super().__init__(
            algorithm="most_popular_for_user",
            item_key_mapping=item_key_mapping,
            user_item_frequency=user_item_frequency,
        )
        # create list out of an ordered dict, sorted by the item frequency descendingly: [(item1, 999), (item2, 998), ...]
        # taking this here as well in case a user only interacted with less than k items
        # self.item_frequency = item_frequency # list(OrderedDict(sorted(item_frequency.items(), key=lambda t: t[1], reverse=True)).items())

        # to make sure that we return a value for all items, put a random small number for the ones
        # np.random.seed(0)
        # self.random_numbers_for_items = {item_key: frequency for (item_key, frequency) in zip(set(self.item_key_mapping.keys()), np.random.rand(self.n_items)*1e-2)}

    def generate_candidates(self, given_items):
        np.random.seed(0)

        candidate_list = np.random.rand(self.n_items) * 1e-2

        return candidate_list

    def rank_candidates(self, user_id, candidate_scores):
        if user_id in self.user_item_frequency:
            for (item_index, frequency) in self.user_item_frequency[user_id].items():
                candidate_scores[self.item_key_mapping[item_index]] += frequency

        return candidate_scores


class ItemCoCountRecommender(Recommender):
    def __init__(self, item_key_mapping, item_co_count_matrix, user_item_frequency):
        super().__init__(
            algorithm="item-co-count",
            item_key_mapping=item_key_mapping,
            user_item_frequency=user_item_frequency,
        )
        self.item_co_count_matrix = item_co_count_matrix

    def generate_candidates(self, given_items):
        np.random.seed(0)

        co_count_sums = np.random.rand(self.n_items) * 1e-2
        target_items = [
            self.item_key_mapping[key]
            if (isinstance(key, str) and key.startswith("product"))
            else key
            for key in given_items
        ]

        for item in target_items:
            # co_count_sums += np.asarray(self.item_co_count_matrix.getrow(item).todense())[0]
            co_count_sums += self.item_co_count_matrix.getrow(item).toarray().flatten()

        # if self.number_of_candidates < self.n_items:
        #     candidate_indices = np.argsort(co_count_sum)[::-1]
        #     candidate_dict = {self.reverse_item_key_mapping[candidate_index]: co_count_sums[candidate_index] for candidate_index in candidate_indices}
        # else:
        #     candidate_dict = {self.reverse_item_key_mapping[candidate_index]: co_count_sum for (candidate_index, co_count_sum) in enumerate(co_count_sums)}

        return co_count_sums

    def rank_candidates(self, user_id, candidate_scores):
        # if self.predict_k < self.n_items return list[:predict_k]; otherwise its equal to self.n_items and I can return all

        return candidate_scores
