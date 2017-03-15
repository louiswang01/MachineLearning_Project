"""
4. Support Vector Machines
"""

from sklearn import svm
import matplotlib.pyplot as plt


def apply_svm(dataset):
    X_train, y_train, X_test, y_test = dataset

    test_scores = []
    train_scores = []

    test_score_sum = 0
    train_score_sum = 0
    for j in range(10):
        clf = svm.SVC(C=1.0, kernel='rbf', max_iter=200)
        clf.fit(X_train, y_train)
        test_score_sum += clf.score(X_test, y_test)
        train_score_sum += clf.score(X_train, y_train)
    test_scores.append(test_score_sum / 10)  # get average score
    train_scores.append(train_score_sum / 10)

    test_score_sum = 0
    train_score_sum = 0
    for j in range(10):
        clf = svm.SVC(C=1.0, kernel='sigmoid', max_iter=200)
        clf.fit(X_train, y_train)
        test_score_sum += clf.score(X_test, y_test)
        train_score_sum += clf.score(X_train, y_train)
    test_scores.append(test_score_sum / 10)  # get average score
    train_scores.append(train_score_sum / 10)

    test_score_sum = 0
    train_score_sum = 0
    for j in range(10):
        clf = svm.SVC(C=1.0, kernel='linear', max_iter=200)
        clf.fit(X_train, y_train)
        test_score_sum += clf.score(X_test, y_test)
        train_score_sum += clf.score(X_train, y_train)
    test_scores.append(test_score_sum / 10)  # get average score
    train_scores.append(train_score_sum / 10)

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title('SVM')
    plt.plot(range(1, 4), train_scores, label='Train')
    plt.plot(range(1, 4), test_scores, label='Test')

    plt.legend()
    # plt.ylim(0.18, 0.62)
    plt.ylabel('Error Score')
    plt.xlabel('Different Kernel')
    plt.show()
