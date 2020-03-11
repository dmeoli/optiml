from ml.losses import cross_entropy
from ml.metrics import accuracy_score
from ml.neural_network.activations import sigmoid, softmax
from ml.neural_network.layers import FullyConnected
from ml.neural_network.neural_network import NeuralNetwork
from optimization.unconstrained.quasi_newton import BFGS
from utils import load_monk

if __name__ == '__main__':
    for i in (1, 2, 3):
        X_train, X_test, y_train, y_test = load_monk(i)
        net = NeuralNetwork(FullyConnected(17, 17, sigmoid),
                            FullyConnected(17, 2, softmax))
        net.fit(X_train, y_train, loss=cross_entropy, optimizer=BFGS, epochs=1000, verbose=False, plot=True)
        print("monk's #" + str(i) + " accuracy: " + str(accuracy_score(net.predict(X_test), y_test)))
