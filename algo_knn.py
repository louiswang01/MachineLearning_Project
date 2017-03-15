"""
5. k-Nearest Neighbors
"""

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def apply_knn(dataset):
    X_train, y_train, X_test, y_test = dataset

    test_scores = []
    train_scores = []

    test_score_sum = 0
    train_score_sum = 0
    for j in range(10):
        neigh = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
        neigh.fit(X_train, y_train)
        test_score_sum += neigh.score(X_test, y_test)
        train_score_sum += neigh.score(X_train, y_train)
    test_scores.append(test_score_sum / 10)  # get average score
    train_scores.append(train_score_sum / 10)

    test_score_sum = 0
    train_score_sum = 0
    for j in range(10):
        neigh = KNeighborsClassifier(n_neighbors=4, weights='uniform', n_jobs=-1)
        neigh.fit(X_train, y_train)
        test_score_sum += neigh.score(X_test, y_test)
        train_score_sum += neigh.score(X_train, y_train)
    test_scores.append(test_score_sum / 10)  # get average score
    train_scores.append(train_score_sum / 10)

    test_score_sum = 0
    train_score_sum = 0
    for j in range(10):
        neigh = KNeighborsClassifier(n_neighbors=6, weights='distance', n_jobs=-1)
        neigh.fit(X_train, y_train)
        test_score_sum += neigh.score(X_test, y_test)
        train_score_sum += neigh.score(X_train, y_train)
    test_scores.append(test_score_sum / 10)  # get average score
    train_scores.append(train_score_sum / 10)

    test_score_sum = 0
    train_score_sum = 0
    for j in range(10):
        neigh = KNeighborsClassifier(n_neighbors=8, weights='distance', n_jobs=-1)
        neigh.fit(X_train, y_train)
        test_score_sum += neigh.score(X_test, y_test)
        train_score_sum += neigh.score(X_train, y_train)
    test_scores.append(test_score_sum / 10)  # get average score
    train_scores.append(train_score_sum / 10)

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title('KNN')
    plt.plot(range(1, 5), train_scores, label='Train')
    plt.plot(range(1, 5), test_scores, label='Test')

    plt.legend()
    # plt.ylim(0.18, 0.62)
    plt.ylabel('Error Score')
    plt.xlabel('Case Number')
    plt.show()
