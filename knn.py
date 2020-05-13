import argparse
import os
import math
import numpy as np


def read_file(file_path):
    if not os.path.exists(file_path):
        raise Exception("File not found: " + file_path)

    X = []
    y = []
    is_header_skipped = False
    with open(file_path, "r") as f:
        for line in f:
            if not is_header_skipped:
                is_header_skipped = True
                continue

            cols = line.split(",")
            cols = [col.strip() for col in cols]
            X.append(np.array(cols[:-1], np.float))
            y.append(np.array(cols[-1:], np.float))

    return np.array(X), np.array(y)


def calculate_euclidean_distance(x_1, x_2):
    diff = x_1 - x_2
    diff_square = np.square(diff)
    return math.sqrt(np.sum(diff_square))


def predict(X, y, k):
    m = X.shape[0]

    y_hat = []  # predictions
    for i in range(m):
        label_distance_pairs = []
        for j in range(m):
            if i != j:
                label_distance_pairs.append((calculate_euclidean_distance(X[i], X[j]), y[j, 0]))

        label_distance_pairs = sorted(label_distance_pairs, key=lambda item: item[0])
        vote_count = {}
        for t in range(k):
            label = label_distance_pairs[t][1]
            vote_count[label] = vote_count.get(label, 0) + 1

        vote_count = sorted(vote_count.items(), key=lambda item: item[1], reverse=True)
        y_hat.append(vote_count[0][0])

    return y_hat


def test(y, predictions):
    m = y.shape[0]
    error = 0.0
    for i in range(m):
        if y[i, 0] != predictions[i]:
            error += 1

    error /= m

    return 1 - error, error


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='knn')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--k', type=int, help='Value for kn number of nearest neighbor', default=10)

    args = parser.parse_args()

    if not args.dataset:
        parser.error('please specify --dataset with corresponding path to dataset')

    if not args.k:
        parser.error('please specify --k parameter')

    X, y = read_file(args.dataset)
    print("Read Training sequence and label set: {0} {1}".format(str(X.shape), str(y.shape)))

    predictions = predict(X, y, args.k)
    accuracy, error = test(y, predictions)

    print("Total accuracy: {0}, error: {1}".format(str(accuracy), str(error)))