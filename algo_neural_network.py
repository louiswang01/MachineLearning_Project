"""
2. Neural Network - Multilayer Perceptron
"""

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def init_ann_params_list():
    params = [{'hidden_layer_sizes': (5,), 'activation': 'logistic',
               'solver': 'sgd', 'learning_rate': 'constant',
               'max_iter': 300, 'momentum': 0.9,
               'learning_rate_init': 0.005},
              {'hidden_layer_sizes': (10,), 'activation': 'logistic',
               'solver': 'sgd', 'learning_rate': 'constant',
               'max_iter': 300, 'momentum': 0.9,
               'learning_rate_init': 0.005},
              {'hidden_layer_sizes': (10,), 'activation': 'logistic',
               'solver': 'sgd', 'learning_rate': 'constant',
               'max_iter': 300, 'momentum': 0.9,
               'learning_rate_init': 0.2},
              {'hidden_layer_sizes': (10,), 'activation': 'logistic',
               'solver': 'sgd', 'learning_rate': 'constant',
               'max_iter': 300, 'momentum': 0.1,
               'learning_rate_init': 0.005},
              {'hidden_layer_sizes': (25,), 'activation': 'logistic',
               'solver': 'sgd', 'learning_rate': 'constant',
               'max_iter': 300, 'momentum': 0.9,
               'learning_rate_init': 0.005},
              {'hidden_layer_sizes': (3,), 'activation': 'logistic',
               'solver': 'sgd', 'learning_rate': 'constant',
               'max_iter': 300, 'momentum': 0.9,
               'learning_rate_init': 0.005}
              ]
    return params


def apply_neural_network(dataset):
    params_list = init_ann_params_list()
    X_train, y_train, X_test, y_test = dataset

    test_scores = []
    train_scores = []

    for params in params_list:
        test_score_sum = 0
        train_score_sum = 0

        for j in range(10):
            mlp = MLPClassifier(**params)
            mlp.fit(X_train, y_train)

            test_score_sum += mlp.score(X_test, y_test)
            train_score_sum += mlp.score(X_train, y_train)
        test_scores.append(test_score_sum / 10)  # get average score
        train_scores.append(train_score_sum / 10)

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title('Neural Network - Multilayer Perceptron')
    plt.plot(range(1, 7), train_scores, label='Train')
    plt.plot(range(1, 7), test_scores, label='Test')

    plt.legend()
    # plt.ylim(0.18, 0.62)
    plt.ylabel('Error Score')
    plt.xlabel('Case Number')
    plt.show()
