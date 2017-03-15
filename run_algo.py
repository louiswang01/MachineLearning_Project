from algo_decision_tree import apply_decision_tree
from algo_neural_network import apply_neural_network
from algo_boosting import apply_adaboosted_dt
from algo_svm import apply_svm
from algo_knn import apply_knn

import numpy as np
from sklearn.model_selection import train_test_split

""" Pre-process data """

def init_data_1():
    # data set for red wine case
    data=np.genfromtxt('redwine_1.csv',delimiter=',',skip_header=True)
    X=data[:,:11]
    y=data[:,-1]

    # divide training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=11)

    return [X_train, y_train, X_test, y_test]


def init_data_2():
    # data set for poker case
    data = np.genfromtxt('poker.csv', delimiter=',', skip_header=True)
    X = data[:, :10]
    y = data[:, -1]

    # divide training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=12)

    return [X_train, y_train, X_test, y_test]


if __name__ == '__main__':

    dataset = init_data_1()
    apply_decision_tree(dataset)
    apply_neural_network(dataset)
    apply_adaboosted_dt(dataset)
    # apply_svm(dataset)
    # apply_knn(dataset)

    dataset = init_data_2()
    apply_decision_tree(dataset)
    apply_neural_network(dataset)
    apply_adaboosted_dt(dataset)
    # apply_svm(dataset)
    # apply_knn(dataset)

