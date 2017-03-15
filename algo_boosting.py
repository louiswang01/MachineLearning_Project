"""
3. Multi-class AdaBoosted Decision Trees
"""
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def apply_adaboosted_dt(dataset):
    X_train, y_train, X_test, y_test = dataset

    test_scores = []
    train_scores = []

    for i in range(1,11):
        test_score_sum = 0
        train_score_sum = 0

        for j in range(10):
            bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6),
                                     n_estimators=i*50, learning_rate=1)

            bdt.fit(X_train, y_train)
            test_score_sum += bdt.score(X_test, y_test)
            train_score_sum += bdt.score(X_train, y_train)
        test_scores.append(test_score_sum / 10)  # get average score
        train_scores.append(train_score_sum / 10)

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title('AdaBoosted Decision Trees')
    plt.plot(range(1, 11), train_scores, label='Train')
    plt.plot(range(1, 11), test_scores, label='Test')

    plt.legend()
    # plt.ylim(0.18, 0.62)
    plt.ylabel('Error Score')
    plt.xlabel('Number of Estimators (hundred)')
    plt.show()
