"""
Load IMDB data and save it to pickle
"""
import pickle as pkl
import os
import random

DATA_ROOT = 'aclImdb'


def get_list_of_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def get_reviews_from_path(path):
    review_list = []
    for file_name in get_list_of_files(path):
        with open(os.path.join(path, file_name), 'r') as f:
            review_list.append(' '.join([line.strip().lower() for line in f.readlines()]))
    return review_list


def load_imdb_from_folder(folder='train'):
    # read positive reviews
    pos_folder_path = os.path.join(DATA_ROOT, folder, 'pos')
    pos_reviews = get_reviews_from_path(pos_folder_path)
    # read negative reviews
    neg_folder_path = os.path.join(DATA_ROOT, folder, 'neg')
    neg_reviews = get_reviews_from_path(neg_folder_path)
    X_text = pos_reviews + neg_reviews
    y = [1] * len(pos_reviews) + [0] * len(neg_reviews)
    # shuffle with pseudo random
    random.seed(666)
    combined = list(zip(X_text, y))
    random.shuffle(combined)
    X_text, y = zip(*combined)
    return X_text, y


if __name__ == '__main__':
    X_text, y = load_imdb_from_folder('train')