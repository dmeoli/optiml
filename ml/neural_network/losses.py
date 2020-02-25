from ml.losses import LossFunction
from ml.metrics import accuracy_score, mean_squared_error


class NeuralNetworkLossFunction(LossFunction):

    def __init__(self, neural_net, loss, verbose=True):
        super().__init__(loss.X, loss.y, loss.regularization_type, loss.lmbda)
        self.neural_net = neural_net
        self.loss = loss
        self.verbose = verbose
        self.loss.predict = lambda X, theta: self.neural_net.forward(X)  # monkeypatch

    def function(self, packed_weights_biases, X, y):
        self.neural_net._unpack(packed_weights_biases)
        loss = self.loss.function(packed_weights_biases, X, y)
        if self.verbose:
            print('Epoch: %i | loss: %.5f | %s: %.2f' %
                  (0, loss, 'acc' if self.neural_net.task is 'classification' else 'mse',
                   accuracy_score(self.neural_net.lb.inverse_transform(y), self.neural_net.predict(X))
                   if self.neural_net.task is 'classification' else mean_squared_error(y, self.neural_net.predict(X))))
        return loss

    def jacobian(self, packed_weights_biases, X, y):
        return self.neural_net._pack(*self.neural_net.backward(self.delta))

    @property
    def delta(self):
        return self.loss.pred - self.loss.target
