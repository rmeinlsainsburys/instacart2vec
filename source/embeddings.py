import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


class Embeddings(object):
    def __init__(self, algorithm, product_key_conversion, with_meta, with_user):
        self.algorithm = algorithm
        self.product_key_conversion = product_key_conversion
        self.with_meta = with_meta
        self.with_user = with_user

    def create_mapping(self):
        self.mapping = {
            item_key: index for index, item_key in enumerate(self.model.wv.index2word)
        }
        self.mapping_back = {
            index: item_key for item_key, index in self.mapping.items()
        }

    def evaluate_embeddings(self, k_neighbors=10):
        y_category_list = []
        y_aisle_list = []

        for key in self.mapping.keys():
            y_category_list.append(self.product_key_to_meta(key).split("\t")[1])
            y_aisle_list.append(self.product_key_to_meta(key).split("\t")[2])

        # K Neighbors Classifier
        k_neighbors_classifier = KNeighborsClassifier(
            n_neighbors=k_neighbors, n_jobs=-1
        )

        category_f1 = self.predict_labels(
            classifier=k_neighbors_classifier,
            x=list(self.embedding_vectors),
            y=y_category_list,
        )
        aisle_f1 = self.predict_labels(
            classifier=k_neighbors_classifier,
            x=list(self.embedding_vectors),
            y=y_aisle_list,
        )

        print(
            f"Category - Micro: {category_f1[0]}, Macro: {category_f1[1]}, Weighted: {category_f1[2]}"
        )
        print(
            f"Aisle    - Micro: {aisle_f1[0]}, Macro: {aisle_f1[1]}, Weighted: {aisle_f1[2]}"
        )

        return category_f1, aisle_f1

    @staticmethod
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

    def product_key_to_meta(self, key):
        assert self.product_key_conversion
        return self.product_key_conversion.get(key, key)

    def product_key_to_name(self, key):
        assert self.product_key_conversion
        return self.product_key_conversion.get(key, key).split("\t")[0]

    def create_embedding_files_for_visualization(self, product_key_conversion):
        """ Create embedding files for visualization """

        target_vectors_filepath = f"visualization/{self.algorithm}_target_vectors.tsv"
        target_metadata_filepath = f"visualization/{self.algorithm}_target_metadata.tsv"

        out_v = open(target_vectors_filepath, "w", encoding="utf-8")
        out_m = open(target_metadata_filepath, "w", encoding="utf-8")

        # Meta File Header
        out_m.write("ProductName\tCategory\tAisle" + "\n")

        for item_key, index in self.mapping.items():
            embedding_vector = self.embedding_vectors[index]
            # META Input
            out_m.write(self.product_key_to_meta(item_key) + "\n")
            out_v.write("\t".join([str(x) for x in embedding_vector]) + "\n")

        out_v.close()
        out_m.close()

    def reduce_dimensions(self, data_loader):
        key_aisle_conversion = {
            aisle_key: aisle_name
            for aisle_name, aisle_key in data_loader.aisle_key_conversion.items()
        }

        key_category_conversion = {
            category_key: category_name
            for category_name, category_key in data_loader.category_key_conversion.items()
        }

        # initial_num_dimensions = 50
        num_dimensions = 2  # final num dimensions (2D, 3D, etc)

        vectors = []
        labels = []
        categories = []
        aisles = []
        for item_key, index in self.mapping.items():
            vectors.append(self.embedding_vectors[index])
            labels.append(self.product_key_to_name(item_key))
            categories.append(self.product_key_to_meta(item_key).split("\t")[1])
            aisles.append(self.product_key_to_meta(item_key).split("\t")[2])

        if self.with_meta:
            for metadata_key, embedding_vector in self.model_metadata_vectors.items():
                vectors.append(embedding_vector)
                labels.append(metadata_key)
                if metadata_key.startswith("category_"):
                    categories.append(key_category_conversion[metadata_key])
                    aisles.append("None")
                else:
                    categories.append("None")
                    aisles.append(key_aisle_conversion[metadata_key])

        # convert both lists into numpy vectors for reduction
        vectors = np.asarray(vectors)
        labels = np.asarray(labels)
        categories = np.asarray(categories)
        aisles = np.asarray(aisles)

        # Initial Reduction
        # print("Initial Reduction...")
        # pca = PCA(n_components=initial_num_dimensions)
        # vectors_pca = pca.fit_transform(vectors)

        # randomly sample data to run quickly
        # rows = np.arange(len(vectors))
        # np.random.shuffle(rows)
        n_select = len(vectors)  # 10000

        # reduce dimensionality using t-SNE
        print("Reduction via T-SNE...")
        tsne = TSNE(
            n_components=num_dimensions,
            verbose=1,
            perplexity=50,
            n_iter=1000,
            learning_rate=10,
            random_state=0,
            n_jobs=-1,
        )
        vectors_tsne = tsne.fit_transform(vectors[:n_select])

        labels = labels[:n_select]
        categories = categories[:n_select]
        aisles = aisles[:n_select]

        x_vals = [v[0] for v in vectors_tsne]
        y_vals = [v[1] for v in vectors_tsne]

        return x_vals, y_vals, labels, categories, aisles

    def plot_with_matplotlib(
        self, x_vals, y_vals, labels, colors, metadata, annotate_with=None
    ):
        plt.figure(figsize=(12, 12))
        plt.scatter(x=x_vals, y=y_vals, c=colors, cmap="tab20", s=10)
        # plt.legend(handles=scatter.legend_elements()[0], labels=list(metadata))

        # Label data points
        indices = []
        for i, label in enumerate(labels):
            if label.startswith(annotate_with):
                indices.append(i)
        # selected_indices = random.sample(indices, 25)
        for i in indices:
            plt.annotate(metadata[i], (x_vals[i], y_vals[i]))

        plt.savefig(f"visualization/{self.algorithm}_scatter_plot.png")

    def visualize_embeddings(self, data_loader, annotate_with):
        encoder = LabelEncoder()
        x_vals, y_vals, labels, categories, aisles = self.reduce_dimensions(data_loader=data_loader)

        if annotate_with.startswith("category"):
            category_labels = encoder.fit_transform(categories)
            self.plot_with_matplotlib(
                x_vals=x_vals,
                y_vals=y_vals,
                labels=labels,
                colors=category_labels,
                metadata=categories,
                annotate_with=annotate_with,
            )
        else:
            aisle_labels = encoder.fit_transform(aisles)
            self.plot_with_matplotlib(
                x_vals=x_vals,
                y_vals=y_vals,
                labels=labels,
                colors=aisle_labels,
                metadata=aisles,
                annotate_with=annotate_with,
            )

        
