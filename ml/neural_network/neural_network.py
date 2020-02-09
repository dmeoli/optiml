import matplotlib.pyplot as plt
import numpy as np

from ml.neural_network.activations import Tanh, Sigmoid, SoftMax
from ml.neural_network.dataloader import DataLoader
from ml.initializers import RandomUniform, Constant, Zeros, GlorotUniform
from ml.neural_network.layers import Dense, Conv2D, MaxPool2D, Flatten
from ml.neural_network.losses import SigmoidCrossEntropy, SparseSoftMaxCrossEntropyWithLogits, MSE
from ml.neural_network.metrics import accuracy
from ml.neural_network.model import Model
from ml.neural_network.optimizers import Adam


class Net(Model):
    def __init__(self):
        super().__init__()
        w_init = GlorotUniform()
        b_init = Constant(0.1)

        self.seq_layers = self.sequential(
            Dense(4, 4, Tanh(), w_init, b_init),
            Dense(4, 4, Tanh(), w_init, b_init),
            Dense(4, 1, Sigmoid()))

    def forward(self, x):
        return self.seq_layers.forward(x)


class CNN(Model):
    def __init__(self):
        super().__init__()
        self.seq_layers = self.sequential(
            Conv2D(1, 6, (5, 5), (1, 1), "same", channels_last=True),  # => [n,28,28,6]
            MaxPool2D(2, 2),  # => [n, 14, 14, 6]
            Conv2D(6, 16, 5, 1, "same", channels_last=True),  # => [n,14,14,16]
            MaxPool2D(2, 2),  # => [n,7,7,16]
            Flatten(),  # => [n,7*7*16]
            Dense(7 * 7 * 16, 10))

    def forward(self, x):
        return self.seq_layers.forward(x)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    x, y = load_iris(return_X_y=True)
    y = y[:, np.newaxis]

    net = Net()
    opt = Adam(net.params, lr=0.1)
    loss_fn = SigmoidCrossEntropy()

    for epoch in range(30):
        o = net.forward(x)
        loss = loss_fn(o, y)
        net.backward(loss)
        opt.step()
        acc = accuracy(o.data > 0.5, y)
        print("Epoch: %i | loss: %.5f | acc: %.2f" % (epoch, loss.data, acc))

    x_test = x
    y_test = y.ravel()
    print(net.forward(x_test).data.ravel(), "\n", y_test)

    # REGRESSION
    # x = np.linspace(-1, 1, 200)[:, None]  # [batch, 1]
    # y = x ** 2 + np.random.normal(0., 0.1, (200, 1))  # [batch, 1]
    #
    # net = Net()
    # opt = Adam(net.params, lr=0.1)
    # loss_fn = MSE()
    #
    # for step in range(100):
    #     o = net.forward(x)
    #     loss = loss_fn(o, y)
    #     net.backward(loss)
    #     opt.step()
    #     print("Step: %i | loss: %.5f" % (step, loss.data))
    #
    # net.save("./params.pkl")
    # net1 = Net()
    # net1.restore("./params.pkl")
    # o2 = net1.forward(x)
    #
    # plt.scatter(x, y, s=20)
    # plt.plot(x, o2.data, c="red", lw=3)
    # plt.show()
    #
    # # CNN
    # f = np.load('./mnist.npz')
    # train_x, train_y = f['x_train'][:, :, :, None], f['y_train'][:, None]
    # test_x, test_y = f['x_test'][:2000][:, :, :, None], f['y_test'][:2000]
    #
    # # from keras.datasets import mnist
    # #
    # # (train_x, train_y), (test_x, test_y) = mnist.load_data()
    #
    # train_loader = DataLoader(train_x, train_y, batch_size=64)
    #
    # cnn = CNN()
    # opt = Adam(cnn.params, 0.001)
    # loss_fn = SparseSoftMaxCrossEntropyWithLogits()
    #
    # for epoch in range(300):
    #     bx, by = train_loader.next_batch()
    #     by_ = cnn.forward(bx)
    #     loss = loss_fn(by_, by)
    #     cnn.backward(loss)
    #     opt.step()
    #     if epoch % 50 == 0:
    #         ty_ = cnn.forward(test_x)
    #         acc = accuracy(np.argmax(ty_.data, axis=1), test_y)
    #         print("Epoch: %i | loss: %.3f | acc: %.2f" % (epoch, loss.data, acc))

    # from ml.dataset import DataSet
    # from ml.validation import grade_learner, err_ratio
    #
    # iris_tests = [([[5.0, 3.1, 0.9, 0.1]], 0),
    #               ([[5.1, 3.5, 1.0, 0.0]], 0),
    #               ([[4.9, 3.3, 1.1, 0.1]], 0),
    #               ([[6.0, 3.0, 4.0, 1.1]], 1),
    #               ([[6.1, 2.2, 3.5, 1.0]], 1),
    #               ([[5.9, 2.5, 3.3, 1.1]], 1),
    #               ([[7.5, 4.1, 6.2, 2.3]], 2),
    #               ([[7.3, 4.0, 6.1, 2.4]], 2),
    #               ([[7.0, 3.3, 6.1, 2.5]], 2)]
    #
    # iris = DataSet(name='iris')
    # classes = ['setosa', 'versicolor', 'virginica']
    # iris.classes_to_numbers(classes)
    # n_samples, n_features = len(iris.examples), iris.target
    # X, y = np.array([x[:n_features] for x in iris.examples]), \
    #        np.array([x[n_features] for x in iris.examples])
    #
    # nnl = NeuralNetworkLearner(iris, [4]).fit(X, y)
    # print(grade_learner(nnl, iris_tests))
    # print(err_ratio(nnl, X, y))
    #
    # pl = PerceptronLearner(iris).fit(X, y)
    # print(grade_learner(pl, iris_tests))
    # print(err_ratio(pl, X, y))
