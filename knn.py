import argparse
import os
import math
import numpy as np
import random


def read_file(file_path):
    if not os.path.exists(file_path):
        raise Exception("File not found: " + file_path)

    dataset = []
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

            dataset.append(([float(t) for t in cols[:-1]], float(cols[-1])))

    return dataset


def shuffle_and_split_dataset(dataset):
    random.shuffle(dataset)  # Shuffling the dataset
    X, y = zip(*dataset)  # Unpacking list of tuples into two separate lists
    m = len(X)
    d = len(X[0])
    X = np.array(list(X)).reshape(m, d)
    y = np.array(list(y)).reshape(m, 1)

    split_point = int(math.ceil(X.shape[0] * 0.8))
    train_X = X[:split_point, :]
    train_y = y[:split_point, :]
    test_X = X[split_point:, :]
    test_y = y[split_point:, :]

    return train_X, train_y, test_X, test_y


def calculate_euclidean_distance(x_1, x_2):
    diff = x_1 - x_2
    diff_square = np.square(diff)
    return math.sqrt(np.sum(diff_square))


def get_k_nearest_neighbors(train_X, train_y, x, k):
    m = train_X.shape[0]
    neighbors = []
    for j in range(m):
        neighbors.append((calculate_euclidean_distance(train_X[j], x), train_y[j, 0]))

    # Sorting list of tuples on the basis of distance
    neighbors = sorted(neighbors, key=lambda item: item[0])

    # Return only k closest neighbors
    return neighbors[:k]


def predict(train_X, train_y, test_X, k):
    m = test_X.shape[0]

    y_hat = []  # predictions
    for i in range(m):
        k_nearest_neighbors = get_k_nearest_neighbors(train_X, train_y, test_X[i], k)

        vote_count = {}
        for distance, label in k_nearest_neighbors:
            vote_count[label] = vote_count.get(label, 0) + 1

        # Sort labels on the basis of votes in descending order and then get the majority vote label
        vote_count = sorted(vote_count.items(), key=lambda item: item[1], reverse=True)
        y_hat.append(vote_count[0][0])

    return y_hat  # Returning predictions


def calculate_accuracy_error(y, predictions):
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

    k = args.k

    dataset = read_file(args.dataset)
    avg_accuracy = 0.0
    avg_error = 0.0
    for i in range(20):
        train_X, train_y, test_X, test_y = shuffle_and_split_dataset(dataset)

        # print("Read Training size: {0}, Test size: {1}".format(str(train_X.shape), str(test_X.shape)))

        predictions = predict(train_X, train_y, test_X, args.k)
        accuracy, error = calculate_accuracy_error(test_y, predictions)
        avg_accuracy += accuracy
        avg_error += error

        print("Total accuracy: {0}, error: {1}, k: {2}".format(str(accuracy), str(error), str(k)))

    print("Average accuracy: " + str(avg_accuracy/20.0) + " Average Error: " + str(avg_error/20.0) + " " + str(k))