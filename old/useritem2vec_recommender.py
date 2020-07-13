import numpy as np
from gensim.matutils import argsort

from recommender import Recommender


class UserItem2VecRecommender(Recommender):
    def __init__(
        self,
        algorithm,
        item_key_mapping,
        embedding_vectors,
        context_vectors,
        user_vectors,
        alpha=0.5,
    ):
        super().__init__(algorithm=algorithm, item_key_mapping=item_key_mapping)
        self.embedding_vectors = embedding_vectors
        self.context_vectors = context_vectors
        self.user_vectors = user_vectors
        self.alpha = alpha

    def generate_candidates(self, given_items):
        candidate_list = []
        # map the items word to its index
        target_items = [self.item_key_mapping[key] for key in given_items]
        # slice the word vectors array to only keep the relevant items
        item_embeddings = self.embedding_vectors[target_items]

        mean_basket_vector = np.mean(item_embeddings, 0)

        # complementary items need to be calculated via dot product not cosine similarity
        distances = np.dot(self.context_vectors, mean_basket_vector)

        # candidate_indices = np.arange(0, len(distances))
        # candidate_list = [(recommender.reverse_item_key_mapping[index], float(distances[index])) for index in candidate_indices]

        return distances

    def rank_candidates(self, user_id, candidate_scores):
        # ranked_candidate_list = []

        # rank the top items by distance to user (from largest to smallest)
        # candidate_embeddings = [recommender.embedding[product_id] for product_id, _ in candidate_list]
        # candidate_embeddings = np.array(candidate_embeddings)

        user_distances = np.dot(self.embedding_vectors, self.user_vectors[user_id])

        combined_distances = (candidate_scores * (1 - self.alpha)) + (
            user_distances * (self.alpha)
        )

        # candidate_indices = np.arange(0, len(combined_distances))
        # candidate_list = [(recommender.reverse_item_key_mapping[index], float(combined_distances[index])) for index in candidate_indices]
        # ranked_candidates = {product_id: score for (product_id, score) in candidate_list}

        # if want to return the most similar items to user out of all items
        # return [item for item, similarity in self.embedding.most_similar([self.user_vectors[user_id]], topn=k)]
        return combined_distances
