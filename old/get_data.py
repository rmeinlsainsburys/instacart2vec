import numpy as np


def get_data(path_data):
    """
    Split the raw sessions into training sessions for item2vec and a common test set.
        path_data: str
        Path to .npy file with shape (nb_sessions,) containing a list of strings
            ['track_0', 'artist_0', ..., 'track_n', 'artist_n']
        in each column.
    """
    np.random.seed(0)

    # load data
    transactions_raw = np.load(path_data, allow_pickle=True)

    # use (1 st, ..., n-1 th) items from sessions to form the train set (drop last product from basket)
    train_set = [transaction[:-1] for transaction in transactions_raw]
    print(f"Original data set shape: {len(train_set)}")

    # sub-sample 10k sessions, and use (n-1 th, n th) pairs of items from sess_p2v to form the disjoint
    # validaton and test sets
    index = np.random.choice(range(len(train_set)), 500000, replace=False)
    validation_set = np.array(train_set)[index[250000:]].tolist()
    test_set = np.array(train_set)[index[:250000]].tolist()
    print(f"Validation set shape: {len(validation_set)}")
    print(f"Test set shape: {len(test_set)}")

    train_set = np.delete(np.array(train_set), index).tolist()
    print(f"Train set shape: {len(train_set)}")

    return train_set, validation_set, test_set
