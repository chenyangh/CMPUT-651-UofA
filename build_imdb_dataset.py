"""
Load IMDB data and save it to pickle
"""
import pickle as pkl
import os
import random
import itertools
from collections import Counter
import numpy as np


DATA_ROOT = 'aclImdb'

VOCAB_SIZE = 2000
HALF_VAL_SIZE = 1250


def get_reviews_from_path(path):
    def get_list_of_files(_path):
        return [f for f in os.listdir(_path) if os.path.isfile(os.path.join(_path, f))]
    review_list = []
    for file_name in get_list_of_files(path):
        with open(os.path.join(path, file_name), 'r') as f:
            review_list.append(' '.join([line.strip().lower() for line in f.readlines()]))
    return review_list


def build_vocab(X_text, vocab_size=2000):
    # flatten list of lists
    def flat_list(list_of_lists):
        return list(itertools.chain(*list_of_lists))

    # count the occurrences of words
    vocab = Counter(flat_list([text.split() for text in X_text]))

    # sort descending order
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    # take top {vocab_size} words
    vocab = vocab[: vocab_size]

    # build word2id and id2word
    word2id = {}
    id2word = {}
    for idx, (word, _) in enumerate(vocab):
        word2id[word] = idx
        id2word[idx] = word

    return word2id, id2word


def build_bow_feature(X_text, word2id, dim=2000):
    """
    :param text:
    :return:
    """
    X = []
    for text in X_text:
        bow = np.zeros(dim)
        for word in text.split():
            if word in word2id:
                bow[word2id[word]] = 1
        X.append(bow)
    return X


def str_clean(text):
    raise NotImplementedError


def remove_stopwords(text):
    raise NotImplementedError


def load_imdb_text(is_train=True, is_str_clean=False, is_remove_stopwords=False):
    if is_train:
        folder = 'train'
    else:
        folder = 'test'

    # read positive reviews
    pos_folder_path = os.path.join(DATA_ROOT, folder, 'pos')
    pos_reviews = get_reviews_from_path(pos_folder_path)
    # read negative reviews
    neg_folder_path = os.path.join(DATA_ROOT, folder, 'neg')
    neg_reviews = get_reviews_from_path(neg_folder_path)

    if is_train:
        # shuffle
        np.random.shuffle(pos_reviews)
        np.random.shuffle(neg_reviews)
        # get HALF_VAL_SIZE from pos and neg respectively to build val set
        val_pos = pos_reviews[:HALF_VAL_SIZE]
        train_pos = pos_reviews[HALF_VAL_SIZE:]
        val_neg = neg_reviews[:HALF_VAL_SIZE]
        train_neg = neg_reviews[HALF_VAL_SIZE:]

        # build train text
        X_train_text = train_pos + train_neg
        y_train = [1] * len(train_pos) + [0] * len(train_neg)

        # shuffle train data
        np.random.seed(314)
        np.random.shuffle(X_train_text)
        np.random.seed(314)
        np.random.shuffle(y_train)

        # build val text
        X_val_text = val_pos + val_neg
        y_val = [1] * len(val_pos) + [0] * len(val_neg)

        return X_train_text, y_train, X_val_text, y_val
    else:
        # Concat Pos, Neg and make label
        X_text = pos_reviews + neg_reviews
        y = [1] * len(pos_reviews) + [0] * len(neg_reviews)
        return X_text, y


def build_imdb_dataset_with_bow(is_str_clean=False, is_remove_stopwords=False):
    X_train_text, y_train, X_val_text, y_val = load_imdb_text(is_train=True)
    X_test_text, y_test = load_imdb_text(is_train=False)

    # String normalization (optional)
    if is_str_clean:
        X_train_text = [str_clean(text) for text in X_train_text]
        X_val_text = [str_clean(text) for text in X_val_text]
        X_test_text = [str_clean(text) for text in X_test_text]

    # remove stop words (optional)
    if is_remove_stopwords:
        X_train_text = [remove_stopwords(text) for text in X_train_text]
        X_val_text = [remove_stopwords(text) for text in X_val_text]
        X_test_text = [remove_stopwords(text) for text in X_test_text]

    word2id, id2word = build_vocab(X_train_text, vocab_size=VOCAB_SIZE)

    # Build BOW
    X_train = build_bow_feature(X_train_text, word2id, dim=VOCAB_SIZE)
    X_val = build_bow_feature(X_val_text, word2id, dim=VOCAB_SIZE)
    X_test = build_bow_feature(X_test_text, word2id, dim=VOCAB_SIZE)

    with open('imdb_data.pkl', 'bw') as f:
        pkl.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)

    with open('imdb_aux.pkl', 'bw') as f:
        pkl.dump((id2word, word2id), f)


if __name__ == '__main__':
    build_imdb_dataset_with_bow()
