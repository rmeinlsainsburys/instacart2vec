import json
import os
from collections import Counter

import numpy as np
import pandas as pd

import stage


class InstacartDataLoader:
    """
    Instacart Data Loader Class
    """

    def __init__(
        self,
        min_item_transactions=0,
        min_user_transactions=0,
        subset_user=1.0,
        algorithm="word2vec",
        stage=stage.TRAIN,
    ):
        self.min_item_transactions = min_item_transactions
        self.min_user_transactions = min_user_transactions
        self.subset_user = subset_user
        self.algorithm = algorithm
        self.stage = stage

    def create_sentences(self, stage: str):
        # Build train and test set
        if self.algorithm == "word2vec":
            data = self.get_stage_data(stage=stage)
            filepath = f"sentences/{self.algorithm}/{stage}.txt"
            if not os.path.exists(filepath):
                print(
                    f"Creating sentences for the {self.stage} stage for {self.algorithm} in {filepath}"
                )
                with open(filepath, "w") as file:
                    for i, row in data.iterrows():
                        file.write(" ".join(map(str, row["product_id"])) + "\n")
                self.word2vec_dir = filepath

    def __iter__(self):
        sentences_path = f"sentences/{self.algorithm}/"
        assert os.path.exists(sentences_path)
        for fname in os.listdir(sentences_path):
            path = os.path.join(sentences_path, fname)
            if self.stage in path:
                for line in open(path):
                    yield line.split()

    def load_and_preprocess_data(self, subset: bool = False):
        small = ""
        if subset:
            self.min_item_transactions = 500
            self.min_user_transactions = 5
            self.subset_user = 0.1
            small = "small_"

        metadata_path = f"preprocessed_data/{small}product_metadata.csv"
        orders_path = f"preprocessed_data/{small}order_data.csv"

        if (not os.path.exists(metadata_path)) or (not os.path.exists(orders_path)):
            print("Data doesn't exist. Creating new...")

            order_products_train_df = pd.read_csv("data/order_products__train.csv")
            order_products_prior_df = pd.read_csv("data/order_products__prior.csv")
            order_products_df = pd.concat(
                [order_products_train_df, order_products_prior_df]
            )[["order_id", "product_id"]]
            print(
                f"Dimensions of concatenated data frame: {order_products_df.shape} \n"
            )
            del order_products_train_df, order_products_prior_df

            # FILTER FOR MIN USER TRANSACTIONS
            orders_df = pd.read_csv("data/orders.csv")[
                ["order_id", "user_id", "order_number", "eval_set"]
            ]
            # remove the orders belonging to the test set, as we don't have those transactions
            orders_df = orders_df[orders_df.eval_set != "test"]
            # Get the number of times a user has ordered (filtering users with less than min_user_transactions)
            tmp = orders_df["user_id"].value_counts()
            user_count = tmp.reset_index()
            user_count.columns = ["user_id", "count"]
            # Select a random subset of users if subset_user is specified (for rapid prototyping)
            user_count["select"] = (
                np.random.rand(user_count.shape[0]) < self.subset_user
            )
            orders_df = pd.merge(
                orders_df,
                user_count.loc[
                    (user_count["select"])
                    & (user_count["count"] >= self.min_user_transactions)
                ],
                on="user_id",
            )
            print(
                f"Orders Shape after Min User Transactions Filtered: {orders_df.shape}"
            )

            # Load and merge products, departments, aisles
            products = pd.read_csv("data/products.csv")
            departments = pd.read_csv("data/departments.csv")
            products = pd.merge(products, departments, on="department_id")
            aisles = pd.read_csv("data/aisles.csv")
            products = pd.merge(products, aisles, on="aisle_id")

            del departments, aisles

            # FILTER FOR MIN ITEM TRANSACTIONS
            order_products_df = pd.merge(
                order_products_df, orders_df[["order_id"]], on="order_id"
            )
            # Get the number of times a product has been ordered (filtering products with less than min_item_transactions)
            tmp = order_products_df["product_id"].value_counts()
            item_count = tmp.reset_index()
            item_count.columns = ["product_id", "count"]
            # create array of Item_count: # of times this count appears in dataset and sort by item_count
            count_i = np.array(list(Counter(item_count["count"].values).items()))
            count_i = count_i[np.argsort(count_i[:, 0]), :]
            # merge products df with items who have a count >= min_item_transactions
            item_descr = pd.merge(
                products,
                item_count.loc[item_count["count"] >= self.min_item_transactions],
                on="product_id",
            ).sort_values(["count"], ascending=False)

            print(f"Metadata Shape after Min Transactions Filtered: {item_descr.shape}")

            # merge order_products df with product_ids and
            order_products_df = pd.merge(
                order_products_df, item_descr[["product_id"]], on="product_id"
            )
            # map product_id to numbers from 0 - len(item_list)
            item_list = item_descr["product_id"].values
            item_dict = dict(zip(item_list, np.arange(len(item_list))))
            # Mapping product id to numbers between 0 - len(item_list)
            item_descr["product_id"] = item_descr["product_id"].apply(
                lambda x: item_dict[x]
            )

            # merge order_products_df with orders_df and map user_id to numbers from 0 - len(user_list)
            orders_df_full = pd.merge(order_products_df, orders_df, on="order_id")

            del order_products_df, orders_df

            user_list = np.array(list(set(orders_df_full["user_id"].values)))
            user_dict = dict(zip(user_list, np.arange(len(user_list))))
            # apply product mapping using item_dict
            # turn sequence of products into list of products for each order for each user
            orders_df_full = orders_df_full.groupby(
                ["eval_set", "user_id", "order_number"]
            )["product_id"].apply(lambda x: [item_dict[k] for k in x])
            orders_df_full = orders_df_full.reset_index()
            # apply user mapping using user_dict
            orders_df_full["user_id"] = orders_df_full["user_id"].apply(
                lambda x: user_dict[x]
            )
            orders_df_full = orders_df_full.sort_values(["order_number"])

            # Get the indexes for the test set (last transaction) and validation set (second-to-last) transaction
            temp_users = orders_df_full.groupby("user_id")
            test_set_index = []
            validation_set_index = []
            for (k, d) in temp_users:
                if len(d) > 1:
                    test_set_index.append(d.index[-1])
                if len(d) > 2:
                    validation_set_index.append(d.index[-2])
            # np.random.seed(0)

            orders_df_full.loc[:, "eval_set"] = stage.TRAIN
            orders_df_full.loc[validation_set_index, "eval_set"] = stage.VALIDATION
            orders_df_full.loc[test_set_index, "eval_set"] = stage.TEST

            # Sanity check
            print(
                f"Average Basket Size (train): {np.mean(orders_df_full.loc[orders_df_full['eval_set'] == stage.TRAIN]['product_id'].apply(lambda x: len(x)))}"
            )
            print(
                f"Average Basket Size (validation): {np.mean(orders_df_full.loc[orders_df_full['eval_set'] == stage.VALIDATION]['product_id'].apply(lambda x: len(x)))}"
            )
            print(
                f"Average Basket Size (test): {np.mean(orders_df_full.loc[orders_df_full['eval_set'] == stage.TEST]['product_id'].apply(lambda x: len(x)))}"
            )

            item_descr.to_csv(metadata_path, index=False)
            orders_df_full.to_csv(orders_path, index=False)
        else:
            print("Data already exists. Loading...")
            orders_df_full = pd.read_csv(orders_path)
            orders_df_full["product_id"] = orders_df_full["product_id"].apply(
                lambda x: list(set(eval(x)))
            )
            item_descr = pd.read_csv(metadata_path)

        self.orders = orders_df_full
        self.metadata = item_descr
        self._product_key_conversion = self._get_product_key_conversion()
        self.number_of_users = len(self.orders["user_id"].unique())
        self.number_of_items = len(self.metadata["product_id"].unique())

        print(
            f"Loaded orders and metadata, containing {self.number_of_users} users and {self.number_of_items} items."
        )

        del orders_df_full, item_descr

    def get_stage_data(self, stage: str = "train"):
        return self.orders[["user_id", "product_id"]].loc[
            self.orders["eval_set"] == stage
        ]

    def _get_product_key_conversion(self) -> dict:
        product_key_conversion_path = f"preprocessed_data/product_key_conversion.json"

        if not os.path.exists(product_key_conversion_path):
            product_key_conversion = {}

            for index, row in self.metadata.iterrows():
                name = (
                    str(row["product_name"])
                    + "\t"
                    + str(row["department"])
                    + "\t"
                    + str(row["aisle"])
                )
                product_key_conversion.setdefault(row["product_id"], name)

            with open(product_key_conversion_path, "w") as f:
                json.dump(product_key_conversion, f)

        else:
            with open(product_key_conversion_path, "r") as f:
                product_key_conversion = json.load(f)

        return product_key_conversion

    def product_key_to_meta(self, key):
        assert self._product_key_conversion
        return self._product_key_conversion.get(key, key)

    def product_key_to_name(self, key):
        assert self._product_key_conversion
        return self._product_key_conversion.get(key, key).split("\t")[0]

    def generate_user_item_interactions(self):
        n_items = self.number_of_items
        train_data = self.get_stage_data(stage=stage.TRAIN)

        user_transactions_map = {}
        user_item_frequency = {}
        item_frequency = np.zeros(n_items)
        for index, row in train_data.iterrows():
            user_id = row["user_id"]
            items = row["product_id"]
            temp_transactions = user_transactions_map.get(user_id, [])
            temp_transactions.append(index)
            user_transactions_map[user_id] = temp_transactions

            temp_item_frequency = user_item_frequency.get(user_id, {})
            for item in items:
                temp_item_frequency[item] = temp_item_frequency.get(item, 0) + 1
                item_frequency[item] += 1
            user_item_frequency[user_id] = temp_item_frequency

        self.user_transactions_map = user_transactions_map
        self.user_item_frequency = user_item_frequency
        self.item_frequency = item_frequency
