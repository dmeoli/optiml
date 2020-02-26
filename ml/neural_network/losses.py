from ml.losses import LossFunction
from ml.metrics import accuracy_score, mean_squared_error
from ml.regularizers import l2


class NeuralNetworkLossFunction(LossFunction):

    def __init__(self, neural_net, loss, regularizer=l2, lmbda=0.01, verbose=True):
        super().__init__(loss.X, loss.y)
        self.neural_net = neural_net
        self.loss = loss
        self.regularizer = regularizer
        self.lmbda = lmbda
        self.verbose = verbose

    def function(self, packed_weights_biases, X, y):
        self.neural_net._unpack(packed_weights_biases)
        self.y_pred = self.neural_net.forward(X)
        loss = self.loss.function(self.y_pred, y) + self.regularizer(packed_weights_biases, self.lmbda) / X.shape[0]
        if self.verbose:
            print('Epoch: %i | loss: %.5f | %s: %.2f' %
                  (0, loss, 'acc' if self.neural_net.task is 'classification' else 'mse',
                   accuracy_score(self.neural_net.lb.inverse_transform(y), self.neural_net.predict(X))
                   if self.neural_net.task is 'classification' else mean_squared_error(y, self.neural_net.predict(X))))
        return loss

    def jacobian(self, packed_weights_biases, X, y):
        return self.neural_net._pack(*self.neural_net.backward(self.delta(y)))

    def delta(self, y_true):
        return self.y_pred - y_true
