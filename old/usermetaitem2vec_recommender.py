import numpy as np
from gensim.matutils import argsort

from recommender import Recommender


class UserMetaItem2VecRecommender(Recommender):
    def __init__(self, embedding, context_vectors, user_vectors):
        super().__init__(algorithm="user-meta-item2vec")
        self.embedding = embedding
        self.context_vectors = context_vectors
        self.user_vectors = user_vectors

    def generate_candidates(self, given_items, number_of_candidates=25):
        candidate_list = []
        # First get the top complementary items
        item_embeddings = [
            self.embedding[key] for key in given_items if key in self.embedding
        ]

        if len(item_embeddings) > 0:
            mean_basket_vector = np.mean(item_embeddings, 0)

            # complementary items need to be calculated via dot product not cosine similarity
            distances = np.dot(self.context_vectors, mean_basket_vector)
            # sorted distances from largest to smallest
            candidate_indices = argsort(
                distances, topn=number_of_candidates, reverse=True
            )

            # to get the distance: float(dists[item])
            candidate_list = [
                self.embedding.index2word[item] for item in candidate_indices
            ]
            filtered_candidate_list = [
                candidate
                for candidate in candidate_list
                if candidate.startswith("product_")
            ]

        return filtered_candidate_list

    def rank_candidates(self, user_id, k, candidate_list):
        ranked_candidate_list = []

        if len(candidate_list) > 0:
            # create an index for the candidate list so that we keep track of its sorting when re-ranking based on the user vector
            candidates_index = {i: item for (i, item) in enumerate(candidate_list)}

            # then rank the top items by distance to user (from largest to smallest)
            candidate_embeddings = [
                self.embedding[key] for key in candidate_list if key in self.embedding
            ]
            candidate_embeddings = np.array(candidate_embeddings)

            user_distances = np.dot(candidate_embeddings, self.user_vectors[user_id])
            ranked_candidate_indices_by_user = argsort(
                user_distances, topn=k, reverse=True
            )

            # use the index provided by the candidate generation stage
            ranked_candidate_list = [
                candidates_index[item] for item in ranked_candidate_indices_by_user
            ]

        # if want to return the most similar items to user out of all items
        # return [item for item, similarity in self.embedding.most_similar([self.user_vectors[user_id]], topn=k)]
        return ranked_candidate_list
