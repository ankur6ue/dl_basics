import numpy as np
import os
import matplotlib.pyplot as plt


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


def draw_movie_frame(net, frame_num):
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
    plt.savefig(script_dir + "/../movie" + "/file%04d.png" % frame_num)