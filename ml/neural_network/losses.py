from ml.losses import LossFunction


class NeuralNetworkLossFunction(LossFunction):

    def __init__(self, neural_net, loss):
        super().__init__(loss.X, loss.y, loss.regularization_type, loss.lmbda)
        self.neural_net = neural_net
        self.loss = loss

    def function(self, packed_weights_bias, X, y):
        self.neural_net._unpack(packed_weights_bias)
        self.loss.predict = lambda *args: self.neural_net.forward(self.neural_net.activations)[-1]  # monkeypatch
        return self.loss.function(self.neural_net.weights, X, y)

    def jacobian(self, packed_weights_bias, X, y):
        return self.neural_net._pack(*self.neural_net.backward(
            X, y, self.neural_net.forward(self.neural_net.activations),
            self.neural_net.deltas, self.neural_net.weight_grads, self.neural_net.bias_grads))
