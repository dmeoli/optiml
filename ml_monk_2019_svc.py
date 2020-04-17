from sklearn.svm import SVC as SKLSVC

from ml.svm import SVC
from utils import load_monk

if __name__ == '__main__':
    for i, k in ((1, 'poly'), (2, 'poly'), (3, 'rbf')):
        X_train, X_test, y_train, y_test = load_monk(i)

        svc = SVC(kernel=k)
        svc.fit(X_train, y_train)
        print(f'custom monk #{i} accuracy: {svc.score(X_test, y_test)}')

        svc = SKLSVC(kernel=k)
        svc.fit(X_train, y_train)
        print(f'sklearn monk #{i} accuracy: {svc.score(X_test, y_test)}')

        print()
