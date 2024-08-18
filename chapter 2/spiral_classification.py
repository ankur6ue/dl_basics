
import numpy as np
import matplotlib.pyplot as plt
import argparse
from src.utils import create_dataset, draw_movie_frame
from src.net import Net, Loss, Optimizer

np.random.seed(seed=1)


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
    net = Net(D, H, K, nH=0)
    optimizer = Optimizer(net, step_size, reg)
    loss = Loss(net)

    # initialize parameters randomly
    num_examples = X.shape[0]
    # number of samples in each batch
    B = (int)(num_examples / 2)
    num_iter = 1200

    for i in range(num_iter):
        o = net.forward(X)
        l = loss.cross_entropy_loss(o, y)
        reg_loss = net.regularization_loss()
        l = l + reg*reg_loss
        loss.backward()
        optimizer.update()
        predicted_class = np.argmax(o, axis=1)
        # write every 10th frame to disk, to convert into a movie later
        # if i % 10 == 0:
        #    draw_movie_frame(net, i)

            # print('training time: %.2f' % execution_time)
        acc = np.mean(predicted_class == y)
        print(f"training iteration: {i}, loss: {l}, training accuracy: {acc}")


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
