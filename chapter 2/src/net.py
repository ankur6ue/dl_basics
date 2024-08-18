import numpy as np
from numpy import linalg as LA

# Fully connected layer
class Fc_layer(object):
    def __init__(self, in_, out_, ps=0, name=None):
        self.W = 0.01 * np.random.randn(in_, out_)
        self.b = np.zeros((1, out_))
        self.ps = ps
        self.name = name

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dl_do):
        self.dl_dw = np.dot(self.X.T, dl_do)
        self.dl_db = np.sum(dl_do, axis=0, keepdims=True)
        # derivative of the output wrt the input to this layer, which is needed
        # by the preceeding layer in backprop
        dl_di = np.dot(dl_do, self.W.T)
        return dl_di

    def l2_norm(self):
        return LA.norm(self.W)

    def update(self, step_size, reg):
        dW = self.dl_dw + reg * self.W
        db = self.dl_db
        self.W += -step_size * dW
        self.b += -step_size * db


class Relu(object):
    def forward(self, X):
        # we need to keep X because we need it in the backward pass!
        self.X = X
        return np.maximum(0, X)

    def backward(self, dl_do):
        # RELU simply needs to backpropagate gradient
        dl_do[self.X <= 0] = 0
        return dl_do

    def l2_norm(self):
        return 0

    def update(self, step_size=0, reg=0):
        pass


class Softmax(object):
    def forward(self, X):
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def l2_norm(self):
        return 0

    def update(self, step_size=0, reg=0):
        pass


class Loss(object):
    def __init__(self, net):
        self.net = net

    def cross_entropy_loss(self, output, labels):
        self.output = output
        self.labels = labels
        p = self.output
        n = p.shape[0]
        return -np.sum(np.log(p[range(n), labels])) / n

    def backward(self):
        p = self.output
        n = p.shape[0] # batch size
        dp = np.copy(p)
        dp[np.arange(n), self.labels] -= 1
        dp /= n # this is the derivative of the loss wrt the input of the softmax layer. Now backprop through
        # the rest of the net
        self.net.backward(dp)


class Net(object):
    def __init__(self, D, H, K, nH=0):
        self.layers = []

        input = Fc_layer(D, H, 0, 'input')
        output = Fc_layer(H, K, 0, 'output')
        self.layers.append(input)
        self.layers.append(Relu())
        for l in range(0, nH):
            h1 = Fc_layer(H, H, 'hidden_layer' + l + 1)
            self.layers.append(h1)
            self.layers.append(Relu())
        self.layers.append(output)
        self.layers.append(Softmax())

    def add(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, X):
        y = X
        for layer in self.layers:
            y = layer.forward(y)
            # return the output of the network
        return y

    def backward(self, do):
        # derivative of loss against output of current layer
        dl_do = do
        for layer in reversed(self.layers[0:-1]):
            dl_do = layer.backward(dl_do)
        return dl_do

    def update(self):
        for layer in self.layers:
            layer.update()

    def regularization_loss(self):
        l2_norm = 0
        for layer in self.layers:
            l2_norm = l2_norm + layer.l2_norm()
        return l2_norm

class Optimizer(object):
    def __init__(self, net, step_size, reg):
        self.net = net
        self.step_size = step_size
        self.reg = reg

    def update(self):
        for layer in self.net.layers:
            layer.update(self.step_size, self.reg)
