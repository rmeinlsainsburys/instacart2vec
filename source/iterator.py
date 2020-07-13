import os

from gensim.models.doc2vec import TaggedDocument


class DatasetIterator(object):
    def __init__(self, data, with_user):
        self.data = data
        self.with_user = with_user

    def __iter__(self):
        for transaction in self.data:
            # user id is always first in list, then all the purchased items
            user_id = transaction[0]
            items = transaction[1:]
            if self.with_user:
                # user is always first: transaction[0] so feed it as tags, items are transaction[1:]
                yield TaggedDocument(words=items, tags=[user_id])
            else:
                yield items


class FileIterator(object):
    def __init__(self, data, algorithm, with_user):
        self.algorithm = algorithm
        self.with_user = with_user
        self.data_filepath = self.create_file_iterator(data)

    def create_file_iterator(self, data):
        # Build train set
        filepath = f"sentences/{self.algorithm}/train.txt"
        print(f"Creating file to iterate for {self.algorithm} in {filepath}")
        with open(filepath, "w") as file:
            for transaction in data:
                if self.with_user:
                    items = transaction
                else:
                    # user id is always first in list, then all the purchased items
                    user_id = transaction[0]
                    items = transaction[1:]

                if len(items) > 0:
                    file.write(" ".join(map(str, items)) + "\n")

        return filepath

    def __iter__(self):
        assert os.path.exists(self.data_filepath)
        for line in open(self.data_filepath):
            transaction = line.split()
            if self.with_user:
                # user is always first: transaction[0] so feed it as tags, items are transaction[1:]
                yield TaggedDocument(words=transaction[1:], tags=[transaction[0]])
            else:
                yield transaction
