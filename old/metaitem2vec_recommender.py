import numpy as np
from gensim.matutils import argsort

from recommender import Recommender


class MetaItem2VecRecommender(Recommender):
    def __init__(self, item_key_mapping, embedding_vectors, context_vectors):
        super().__init__(algorithm="meta-item2vec", item_key_mapping=item_key_mapping)
        self.embedding_vectors = embedding_vectors
        self.context_vectors = context_vectors

    def generate_candidates(self, given_items):
        candidate_list = []
        # item_embeddings = [self.embedding[key] for key in given_items]
        # map the items word to its index
        target_items = [self.item_key_mapping[key] for key in given_items]
        # slice the word vectors array to only keep the relevant items
        item_embeddings = self.embedding_vectors[target_items]

        mean_basket_vector = np.mean(item_embeddings, 0)

        # complementary items need to be calculated via dot product not cosine similarity
        distances = np.dot(self.context_vectors, mean_basket_vector)

        # sorted distances from largest to smallest, sort only when we're not in evaluation to save time
        # if self.number_of_candidates < self.n_items:
        #    candidate_indices = argsort(distances, topn=self.number_of_candidates, reverse=True)
        # else:
        #    candidate_indices = np.arange(0, len(distances))

        # candidate_list = [(self.reverse_item_key_mapping[index], float(distances[index])) for index in candidate_indices if index in self.reverse_item_key_mapping]

        # filter out instances of category, shouldn't need this as the mapping already filters out the non products (categories)
        # filtered_candidate_list = [(product_id, score) for (product_id, score) in candidate_list if product_id.startswith('product_')]

        return distances

    def rank_candidates(self, user_id, candidate_scores):
        # if self.predict_k < self.n_items do candidate_list[:predict_k]; otherwise its equal to self.n_items and I can apply it to all
        # ranked_candidates = {product_id: score for (product_id, score) in candidate_list[:self.predict_k]}

        return candidate_scores
