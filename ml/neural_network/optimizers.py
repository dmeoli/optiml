import numpy as np


class Optimizer:

    def __init__(self, params, l_rate):
        self.params = params
        self.lr = l_rate
        self.vars = []
        self.grads = []
        for layer_p in self.params.values():
            for p_name in layer_p['vars'].keys():
                self.vars.append(layer_p['vars'][p_name])
                self.grads.append(layer_p['grads'][p_name])

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, l_rate):
        super().__init__(params, l_rate)

    def step(self):
        for var, grad in zip(self.vars, self.grads):
            var -= self.lr * grad


class Momentum(Optimizer):
    def __init__(self, params, l_rate=0.001, momentum=0.9):
        super().__init__(params, l_rate)
        self.momentum = momentum
        self.mv = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, mv in zip(self.vars, self.grads, self.mv):
            dv = self.lr * grad
            mv[:] = self.momentum * mv + dv
            var -= mv


class AdaGrad(Optimizer):
    def __init__(self, params, l_rate=0.01, eps=1e-06):
        super().__init__(params, l_rate)
        self.eps = eps
        self.v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, v in zip(self.vars, self.grads, self.v):
            v += np.square(grad)
            var -= self.lr * grad / np.sqrt(v + self.eps)


class Adadelta(Optimizer):
    def __init__(self, params, l_rate=1., rho=0.9, eps=1e-06):
        super().__init__(params, l_rate)
        self.rho = rho
        self.eps = eps
        self.m = [np.zeros_like(v) for v in self.vars]
        self.v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        # according to: https://pytorch.org/docs/stable/_modules/torch/optim/adadelta.html#Adadelta
        for var, grad, m, v in zip(self.vars, self.grads, self.m, self.v):
            v[:] = self.rho * v + (1. - self.rho) * np.square(grad)
            delta = np.sqrt(m + self.eps) / np.sqrt(v + self.eps) * grad
            var -= self.lr * delta
            m[:] = self.rho * m + (1. - self.rho) * np.square(delta)


class RMSProp(Optimizer):
    def __init__(self, params, l_rate=0.01, alpha=0.99, eps=1e-08):
        super().__init__(params, l_rate)
        self.alpha = alpha
        self.eps = eps
        self.v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, v in zip(self.vars, self.grads, self.v):
            v[:] = self.alpha * v + (1. - self.alpha) * np.square(grad)
            var -= self.lr * grad / np.sqrt(v + self.eps)


class Adam(Optimizer):
    def __init__(self, params, l_rate=0.01, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params, l_rate)
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(v) for v in self.vars]
        self.v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        b1, b2 = self.betas
        b1_crt, b2_crt = b1, b2
        for var, grad, m, v in zip(self.vars, self.grads, self.m, self.v):
            m[:] = b1 * m + (1. - b1) * grad
            v[:] = b2 * v + (1. - b2) * np.square(grad)
            b1_crt, b2_crt = b1_crt * b1, b2_crt * b2  # bias correction
            m_crt = m / (1. - b1_crt)
            v_crt = v / (1. - b2_crt)
            var -= self.lr * m_crt / np.sqrt(v_crt + self.eps)


class AdaMax(Optimizer):
    def __init__(self, params, l_rate=0.01, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params, l_rate)
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(v) for v in self.vars]
        self.v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        b1, b2 = self.betas
        b1_crt = b1
        for var, grad, m, v in zip(self.vars, self.grads, self.m, self.v):
            m[:] = b1 * m + (1. - b1) * grad
            v[:] = np.maximum(b2 * v, np.abs(grad))
            b1_crt = b1_crt * b1  # bias correction
            m_crt = m / (1. - b1_crt)
            var -= self.lr * m_crt / (v + self.eps)
