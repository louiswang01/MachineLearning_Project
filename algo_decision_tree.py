"""
1. Decision Tree
"""
import matplotlib.pyplot as plt
import numpy as np
import pydotplus
from sklearn import tree

def apply_decision_tree(dataset):
    X_train, y_train, X_test, y_test = dataset

    test_scores=[]
    train_scores = []

    for i in range(1,11):
        test_score_sum = 0
        train_score_sum = 0
        for j in range(10):
            clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
            clf = clf.fit(X_train, y_train)

            # save tree structure to file
            # comment this before handing in
            # dot_data = tree.export_graphviz(clf, out_file=None, feature_names=header,
            #                                 class_names=map(str, range(1, 11)), filled=True, rounded=True)
            # graph = pydotplus.graph_from_dot_data(dot_data)
            # graph.write_pdf("redwine_1.pdf")

            test_score_sum+=clf.score(X_test,y_test)
            train_score_sum+=clf.score(X_train,y_train)
        test_scores.append(test_score_sum/10)     #get average score
        train_scores.append(train_score_sum / 10)

    fig=plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title('Decision Tree')
    plt.plot(range(1,11),train_scores, label='Train')
    plt.plot(range(1, 11), test_scores, label='Test')

    plt.legend()
    # plt.ylim(0.18, 0.62)
    plt.ylabel('Error Score')
    plt.xlabel('Number of Depth')
    plt.show()
