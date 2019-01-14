"""
Data set: the MNIST digits data-set, using the first 60,000 examples and only the examples tagged 0,1,2,3
x = (784,) - a 28X28 matrix ordered as a 784 vector
y in {0,1,2,3}
k = 4
We will implement multi-class SVM with a reduction to the binary problem in two techniques:
1. One VS All (OA):
    We will have k hypotheses, one for each class
2. All Pairs (AP):
    We will have (k choose 2) hypotheses, one for each tuple of two classes
The prediction will be done using a matrix M, a matrix with values {+1, 0, -1} of size (kXl),
where l is k for the 'OA', and k choose 2 for AP.
each row will represent a class, and each column will represent one hypothesis.

Student: Hodaya Koslowsky
"""

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


def hamming_distance(row1, row2):
    distance = 0
    for i, j in zip(row1, row2):
        distance += ((1 - np.sign(i*j)) / 2)
    return distance


def loss_distance(row1, row2):
    distance = 0
    for i, j in zip(row1, row2):
        distance += max(0, 1-i*j)
    return distance


def train(M, l):
    W = np.zeros((l, x_size))
    for hypothesis in range(l):
        for t in range(1, T):
            i = np.random.randint(0, n_examples)
            xi = x[i]
            correct_class = np.int(y[i])
            yi = M[correct_class, hypothesis]
            if yi == 0:
                continue
            eta_t = eta / np.sqrt(t)
            W[hypothesis] = np.multiply(W[hypothesis], 1 - (gamma * eta_t))
            if 1 - (yi*np.dot(W[hypothesis], xi)) > 0:
                W[hypothesis] += np.multiply(eta_t*yi, xi)

    return W


def test(W, M, l, D):
    n_test_examples = x_test.shape[0]
    y_pred = np.subtract(np.zeros(n_test_examples), 1)
    for i in range(n_test_examples):
        xi = x_test[i]
        f_x = np.zeros(l)
        for hypothesis in range(l):
            f_x[hypothesis] = np.dot(W[hypothesis], xi)
        y_hat = -1
        min_distance = np.inf
        for current_class in range(n_classes):
            d = D(M[current_class], f_x)
            if d < min_distance:
                y_hat = current_class
                min_distance = d
        y_pred[i] = y_hat
    return y_pred


def compute_accuracy(real_y, my_y):
    total = real_y.shape[0]
    correct = 0
    for i,j in zip(real_y, my_y):
        if i == j:
            correct += 1
    return correct / total


if __name__ == '__main__':
    mnist = fetch_mldata("MNIST original", data_home="./data")
    X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    x = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
    y = [ey for ey in Y if ey in [0, 1, 2, 3]]
    x, y = shuffle(x, y, random_state=1)

    # np.save('X', x)
    # np.save('Y', y)
    # x = np.load('X.npy')
    # y = np.load('Y.npy')

    x_test = np.loadtxt('x4pred.txt')
    # y_test = np.loadtxt('y_test.txt')

    n_classes = 4
    n_examples = x.shape[0]
    x_size = x.shape[1]

    l_OA = 4
    l_AP = 6
    l_R = 4
    M_OA = np.subtract(np.multiply(np.eye(n_classes, dtype=int), 2), np.ones((n_classes, n_classes), dtype=int))
    M_AP = np.array([[1, 1, 1, 0, 0, 0], [-1, 0, 0, 1, 1, 0], [0, -1, 0, -1, 0, 1], [0, 0, -1, 0, -1, -1]], dtype=int)
    M_R = np.random.randint(-1, 2, (l_R, n_classes))

    eta = 0.1
    gamma = 0.3
    T = 30000

    # One VS All & Hamming
    W1 = train(M_OA, l_OA)
    pred1 = test(W1, M_OA, l_OA, hamming_distance)
    file1 = 'test.onevall.ham.pred'
    np.savetxt(file1, pred1)

    # One VS All & Loss
    W2 = train(M_OA, l_OA)
    pred2 = test(W2, M_OA, l_OA, loss_distance)
    file2 = 'test.onevall.loss.pred'
    np.savetxt(file2, pred2)

    # All Pairs & Hamming
    W3 = train(M_AP, l_AP)
    pred3 = test(W3, M_AP, l_AP, hamming_distance)
    file3 = 'test.allpairs.ham.pred'
    np.savetxt(file3, pred3)

    # All Pairs & Loss
    W4 = train(M_AP, l_AP)
    pred4 = test(W4, M_AP, l_AP, loss_distance)
    file4 = 'test.allpairs.loss.pred'
    np.savetxt(file4, pred4)
    np.save('W', W4)

    # Random & Hamming
    W5 = train(M_R, l_R)
    pred5 = test(W5, M_R, l_R, hamming_distance)
    file5 = 'test.randm.ham.pred'
    np.savetxt(file5, pred5)

    # Random & Loss
    W6 = train(M_R, l_R)
    pred6 = test(W6, M_R, l_R, loss_distance)
    file6 = 'test.randm.loss.pred'
    np.savetxt(file6, pred6)
