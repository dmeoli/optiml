from ml.activations import sigmoid, softmax
from ml.layers import FullyConnected
from ml.losses import cross_entropy
from ml.neural_network import NeuralNetworkClassifier
from optimization.unconstrained.quasi_newton import BFGS
from utils import load_monk

if __name__ == '__main__':
    for i in (1, 2, 3):
        X_train, X_test, y_train, y_test = load_monk(i)
        net = NeuralNetworkClassifier((FullyConnected(17, 17, sigmoid),
                                       FullyConnected(17, 2, softmax)),
                                      loss=cross_entropy, optimizer=BFGS, epochs=100, verbose=True, plot=True)
        net.fit(X_train, y_train, X_test, y_test)
        print(f'monk #{i} accuracy: {net.score(X_test, y_test)}\n')
