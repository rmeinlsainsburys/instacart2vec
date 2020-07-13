import numpy as np
from gensim.matutils import argsort

from recommender import Recommender


class Item2VecRecommender(Recommender):
    def __init__(
        self,
        algorithm,
        item_key_mapping,
        user_item_frequency,
        embedding_vectors,
        context_vectors,
        user_vectors=None,
        alpha=0.5,
    ):
        super().__init__(
            algorithm=algorithm,
            item_key_mapping=item_key_mapping,
            user_item_frequency=user_item_frequency,
        )
        self.embedding_vectors = embedding_vectors
        self.context_vectors = context_vectors
        self.user_vectors = user_vectors
        self.alpha = alpha

    def generate_candidates(self, given_items):
        candidate_list = []
        # map the items word to its index
        target_items = [
            self.item_key_mapping[key]
            if (isinstance(key, str) and key.startswith("product"))
            else key
            for key in given_items
        ]
        # slice the word vectors array to only keep the relevant items
        item_embeddings = self.embedding_vectors[target_items]

        mean_basket_vector = np.mean(item_embeddings, 0)

        # complementary items need to be calculated via dot product not cosine similarity
        distances = np.dot(self.context_vectors, mean_basket_vector)

        return distances

    def rank_candidates(self, user_id, candidate_scores):
        if self.user_vectors:
            user_distances = np.dot(self.embedding_vectors, self.user_vectors[user_id])

            candidate_scores = (candidate_scores * (1 - self.alpha)) + (
                user_distances * (self.alpha)
            )

        return candidate_scores
