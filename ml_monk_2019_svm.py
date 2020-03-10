from ml.metrics import accuracy_score
from ml.svm.kernels import *
from ml.svm.svm import SVC
from utils import load_monk

if __name__ == '__main__':
    for i in (1, 2, 3):
        X_train, X_test, y_train, y_test = load_monk(i)
        svc = SVC(kernel=linear_kernel).fit(X_train, y_train)
        print("monk's #" + str(i) + " accuracy: " + str(accuracy_score(svc.predict(X_test), y_test)))
