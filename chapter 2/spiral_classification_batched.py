
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
np.random.seed(seed=1)


def create_dataset(K, N, D=2):
    # Code to generate spiral dataset
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


# Fully connected layer
class Fc_layer(object):
    def __init__(self, in_, out_, reg, steo_size, ps=0, name=None):
        self.W = 0.01 * np.random.randn(in_, out_)
        self.b = np.zeros((1, out_))
        self.ps = ps
        self.name = name

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dl_do):
        self.dl_dw = np.dot(self.X.T, dl_do) + self.reg * self.W
        self.dl_db = np.sum(dl_do, axis=0, keepdims=True)
        # derivative of the output wrt the input to this layer, which is needed
        # by the preceeding layer in backprop
        dl_di = np.dot(dl_do, self.W.T)
        return dl_di

    def update(self):
        dW = self.dl_dw
        db = self.dl_db
        self.W += -self.step_size * dW
        self.b += -self.step_size * db


class Relu(object):
    def forward(self, X):
        # we need to keep X because we need it in the backward pass!
        self.X = X
        return np.maximum(0, X)

    def backward(self, dl_do):
        # RELU simply needs to backpropagate gradient
        dl_do[self.X <= 0] = 0
        return dl_do

    def update(self):
        pass


class Softmax(object):
    def forward(self, X):
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def update(self):
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
        n = p.shape[0]
        dp = np.copy(p)
        dp[np.arange(n), self.labels] -= 1
        dp /= n # this is the derivative of the loss wrt the input of the softmax layer. Now backprop through
        # the rest of the net
        self.net.backward(dp)


class Net(object):
    def __init__(self, D, H, K, step_size, reg, nH=0):
        self.layers = []

        input = Fc_layer(D, H, step_size, reg, 'input')
        output = Fc_layer(H, K, step_size, reg, 'output')
        net.add(input).add(Relu())
        for l in range(0, nH):
            h1 = Fc_layer(H, H, step_size, reg, 'hidden_layer' + l + 1)
            net.add(h1).add(Relu())
        net.add(output).add(Softmax())

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

    def train(self, X, y, loss, num_iter):
        """
        :param X: Input data
        :param y: Output
        :param num_iter: number of training iterations
        :param loss: Loss function to use
        :return: Total training time
        """
        for i in range(num_iter):
            o = self.forward(X)
            l = loss.cross_entropy_loss(o, y)
            loss.backward()
            net.update()
            predicted_class = np.argmax(o, axis=1)
            # write every 10th frame to disk, to convert into a movie later
            if i % 10 == 0:
                draw_movie_frame(net, i)

            # print('training time: %.2f' % execution_time)
            print('training accuracy: %.2f' % (np.mean(predicted_class == y)))


    def update(self):
        for layer in self.layers:
            layer.update()


def create_net(D, H, nH=0):



def draw_movie_frame(net, frame_num):
    import matplotlib.pyplot as plt
    # create array of numbers size 100, from -1 to 1
    x = np.array([-1 + 0.02 * n for n in range(100)])
    y = np.array([-1 + 0.02 * n for n in range(100)])

    # create an array with coordinates of pixels on a 100, 100 frame
    XX = np.empty((0, 2))
    for x_ in x:
        for y_ in y:
            XX = np.vstack((XX, [x_, y_]))

    # Ask the neural net to produce the probability distribution over classes for each pixel on this frame
    o = net.forward(XX)
    # Get the predicted class and reshape into a 100, 100 frame
    predicted_class = np.argmax(o, axis=1).reshape(100, 100)
    # Show the frame and save to a file. We'll use ffmpeg to make a movie out of the frames
    plt.imshow(predicted_class)
    script_dir = os.path.dirname(__file__)
    plt.savefig(script_dir + "/movie" + "/file%04d.png" % frame_num)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a spiral dataset and build a small neural network to identify'
                                                 'points belonging to each spiral')

    parser.add_argument('--N', type=int, default=350, help='Number of points in the spiral dataset')
    parser.add_argument('--D', type=int, default=2, help='Dimensionality of the data')
    parser.add_argument('--K', type=int, default=4, help='Number of classes')
    parser.add_argument('--H', type=int, default=100, help='Size of hidden layer')
    args = parser.parse_args()
    N = args.N
    D = args.D
    K = args.K
    H = args.H
    X, y = create_dataset(K, N)
    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    # Create the Neural Network
    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3  # regularization strength
    net = Net(D, H, K, step_size, reg, nH=0)
    loss = Loss(net)

    # initialize parameters randomly
    num_examples = X.shape[0]
    # number of samples in each batch
    B = (int)(num_examples / 2)
    net = net.train(X, y, loss)

    predicted_class = np.argmax(scores, axis=1)
    print('training time: %.2f' % execution_time)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y[0:B])))

# something interesting:
# a = np.array([1])
# b = a
# a = a + [1] a will be 2, b will point to old a, and will be 1

# a=np.array([1])
# b = a
# a += [1] # both b and a will be [2]!
