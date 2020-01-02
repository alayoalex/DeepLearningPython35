import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
 
    return images, labels


def viewing_each_character_sample(x_samples, y_samples):
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()

    for i in range(10):
        img = x_samples[y_samples == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    #plt.savefig('./figures/mnist_all.png', dpi=300)
    plt.show()


def viewing_samples_one_character(x_samples, y_samples):
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = x_samples[y_samples == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    #plt.savefig('./figures/mnist_7.png', dpi=300)
    plt.show()