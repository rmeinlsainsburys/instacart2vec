import copy

import numpy as np
import pandas as pd

from iterator import DatasetIterator, FileIterator


class DataLoader(object):
    def __init__(
        self,
        algorithm,
        small_data,
        with_meta,
        with_user,
        use_file_iterator=True,
        use_np_array=True,
    ):
        self.algorithm = algorithm
        self.small_data = small_data
        self.with_meta = with_meta
        self.with_user = with_user

        print(f"With Metadata: {self.with_meta}")
        print(f"With User: {self.with_user}")

        print("Loading data...")
        self.load_data()

        self.product_key_conversion = self.get_product_key_conversion()
        self.category_key_conversion = self.get_category_key_conversion()
        self.aisle_key_conversion = self.get_aisle_key_conversion()
        
        if self.with_meta:
            self.train_data = self.add_product_metadata(
                transaction_data=self.train_data, add_category=True, add_aisle=True
            )

        if use_file_iterator:
            self.train_data_iterator = FileIterator(
                data=self.train_data, algorithm=self.algorithm, with_user=self.with_user
            )
        else:
            self.train_data_iterator = DatasetIterator(
                data=self.train_data, with_user=self.with_user
            )

        self.user_item_frequency = self.generate_user_item_interactions()

    def load_data(self):
        """ Load data """
        small = ""
        if self.small_data:
            small = "small_"

        metadata_path = f"preprocessed_data/{small}product_metadata.csv"
        train_orders_path = f"preprocessed_data/{small}train_orders.npy"
        validation_orders_path = f"preprocessed_data/{small}validation_orders.npy"
        test_orders_path = f"preprocessed_data/{small}test_orders.npy"

        # orders_path = f"preprocessed_data/{small}order_data.csv"

        self.train_data = np.load(train_orders_path, allow_pickle=True)
        self.validation_data = np.load(validation_orders_path, allow_pickle=True)
        self.test_data = np.load(test_orders_path, allow_pickle=True)
        self.item_metadata = pd.read_csv(metadata_path)

    def generate_user_item_interactions(self):
        user_item_frequency = {}
        for index, test_transaction in enumerate(self.train_data):
            # user id is always first in list, then all the purchased items
            user_id = test_transaction[0]
            items = test_transaction[1:]

            temp_item_frequency = user_item_frequency.get(user_id, {})
            for item in items:
                temp_item_frequency[item] = temp_item_frequency.get(item, 0) + 1
            user_item_frequency[user_id] = temp_item_frequency

        return user_item_frequency

    def get_product_key_conversion(self) -> dict:
        product_key_conversion = {}

        for index, row in self.item_metadata.iterrows():
            name = (
                str(row["product_name"])
                + "\t"
                + str(row["department"])
                + "\t"
                + str(row["aisle"])
            )
            product_key_conversion.setdefault(f'product_{row["product_id"]}', name)

        return product_key_conversion

    def get_category_key_conversion(self) -> dict:
        category_key_conversion = {}

        for index, row in self.item_metadata.iterrows():
            category_key_conversion.setdefault(
                str(row["department"]), f'category_{row["department_id"]}'
            )

        return category_key_conversion

    def category_to_key(self, category):
        assert self.category_key_conversion
        return self.category_key_conversion.get(category, category)

    def get_aisle_key_conversion(self) -> dict:
        aisle_key_conversion = {}

        for index, row in self.item_metadata.iterrows():
            aisle_key_conversion.setdefault(
                str(row["aisle"]), f'aisle_{row["aisle_id"]}'
            )

        return aisle_key_conversion

    def aisle_to_key(self, aisle):
        assert self.aisle_key_conversion
        return self.aisle_key_conversion.get(aisle, aisle)

    def add_product_metadata(
        self, transaction_data, add_category=True, add_aisle=False
    ):
        for index, transaction in enumerate(transaction_data):
            items_with_categories = []

            # user id is always first in list, then all the purchased items
            items_with_categories.append(transaction[0])

            items_without_user = copy.deepcopy(transaction[1:])

            # add product metadata to the item list
            for item in items_without_user:
                items_with_categories.append(item)
                if add_category:
                    category_key = self.category_to_key(
                        self.product_key_conversion.get(item).split("\t")[1]
                    )
                    items_with_categories.append(category_key)
                if add_aisle:
                    aisle_key = self.aisle_to_key(
                        self.product_key_conversion.get(item).split("\t")[2]
                    )
                    items_with_categories.append(aisle_key)

            transaction_data[index] = items_with_categories

        return transaction_data
