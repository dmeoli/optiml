import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from ml.learning import BinarySVM
from utils import rbf_kernel


def data_generator(data_type='separable', n=200):
    # data_type = 'soft'
    # data_type = 'circle'
    n_pos, n_neg = n, n
    if data_type == 'circle':
        # positive examples: near origin
        pos = np.random.random((n_pos, 2)) - 0.5
        # negative examples: near a circle or radius 3.0
        angle, radius = 2 * np.pi * np.random.random(n_neg), 1.5
        neg = radius * np.c_[np.cos(angle), np.sin(angle)] + (np.random.random((n_pos, 2)) - 0.5)
    elif data_type == 'soft':
        pos = np.random.random((n_pos, 2)) - 0.25
        neg = np.random.random((n_pos, 2)) + 0.25
    else:
        pos = np.random.random((n_pos, 2)) + 2.
        neg = np.random.random((n_pos, 2)) + 3.
    return pos, neg, n_pos, n_neg, data_type


def demo_test(clf, test_data=data_generator()):
    pos, neg, n_pos, n_neg, data_type = test_data
    x, y = np.r_[pos, neg], np.r_[np.ones(n_pos), -np.ones(n_neg)]
    clf.fit(x, y)
    predicted_y = clf.predict(x)
    # print predicted_y
    target_test = np.array([1.0] * n_pos + [-1.0] * n_neg)
    # target_predict = [clf.predict(sample) for sample in data_train]
    print("\n\nClassification report for classifier %s:\n%s\n"
          % (clf, classification_report(target_test, predicted_y)))
    print("Confusion matrix:\n%s" % confusion_matrix(target_test, predicted_y))
    plt.scatter(pos[:, 0], pos[:, 1], marker='o')
    plt.scatter(neg[:, 0], neg[:, 1], marker='x')
    # wx = np.arange(-radius-0.5, radius+0.5, 0.05)
    if data_type == 'separable' or data_type == 'soft':
        if isinstance(clf, BinarySVM):
            w, b = clf.w, clf.b
        else:
            w = clf.coef_[0]
            b = clf.intercept_[0]
        wx = np.arange(np.min(x[:, 0]) - 0.5, np.max(x[:, 0]) + 0.5, 0.05)
        wy = (-b - w[0] * wx) / w[1]
        plt.plot(wx, wy)
        # r1, r2 = min(min(wx), min(wy)), max(max(wx), max(wy))
        # plt.xlim([r1, r2])
        # plt.ylim([r1, r2])
        plt.show()


if __name__ == "__main__":
    # test_type = 'separable'
    # test_type = 'soft'
    test_type = 'circle'
    data = data_generator(test_type, n=500)
    pos, neg = data[0], data[1]
    plt.scatter(pos[:, 0], pos[:, 1], marker='o')
    plt.scatter(neg[:, 0], neg[:, 1], marker='x')
    plt.show()
    if test_type != 'circle':
        demo_test(BinarySVM(), data)
    else:
        demo_test(BinarySVM(kernel=rbf_kernel), data)
