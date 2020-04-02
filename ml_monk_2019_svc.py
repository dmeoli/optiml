from ml.kernels import polynomial_kernel
from ml.svm import SVC
from optimization.constrained.projected_gradient import ProjectedGradient
from utils import load_monk

if __name__ == '__main__':
    for i in (1, 2, 3):
        X_train, X_test, y_train, y_test = load_monk(i)
        svc = SVC(kernel=polynomial_kernel, optimizer=ProjectedGradient, verbose=False).fit(X_train, y_train)
        print(f'monk #{i} accuracy: {svc.score(X_test, y_test)}')
