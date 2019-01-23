import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn import datasets
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR


def get_data(n_points, data_type, torch_var=True):
    """
     Returns a synthetic dataset.
    """
    if data_type == 'line':
        dx = np.random.randn(n_points, 1)
        dx /= dx.max()
        dy = dx
    elif data_type == 'par':
        dx = np.random.uniform(low=-1, high=1, size=(n_points, 1))
        # dx /= dx.max()
        dy = dx**2
    elif data_type == 'spiral':
        n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
        dx = -np.cos(n)*n + np.random.rand(n_points,1)
        dy = np.sin(n)*n + np.random.rand(n_points,1)
        dx /= dx.max()
        dy /= dy.max()
    else:
        print('Data type not supported.')
        return
    if torch_var:
        data = torch.from_numpy(np.hstack((dx,dy))).float()
    else:
        data = dx,dy
    return data


def plot_data(X, title='Generator output'):
    plt.plot(X[:, 0], X[:, 1], '.')
    plt.title(title)
    plt.legend()
    plt.show()


class Generator(nn.Module):
    def __init__(self, input_size, output_size, drop_p=0.5):
        super(Generator, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(512, output_size),
            nn.Tanh()
        )

        self.drop_layer = nn.Dropout(p=drop_p)

    def forward(self, x):
        x = self.hidden0(x)
        x = self.drop_layer(self.hidden1(x))
        x = self.drop_layer(self.hidden2(x))
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()

        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


def create_uniform_data(n_examples, n_points=None):
    if n_points is None:
        uni = torch.rand(n_examples)
    else:
        uni = torch.rand(n_examples, n_points)
    uni = uni*2 - 1
    # the data is a uniform sample from the range [-1,1]
    return uni


def plot_generated(G, seed_size, data_type, epoch, n_points=1000):
    G.eval()
    seed = create_uniform_data(n_points, seed_size)
    points = G(seed)
    xs = points[:,0].tolist()
    ys = points[:,1].tolist()
    real_x, real_y = get_data(n_points, data_type, torch_var=False)
    plt.plot(real_x, real_y, '.', color='black')
    plt.plot(xs, ys, '.', color='red', linewidth=0.8)
    plt.legend(['Original', 'Generated'])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title('Generated examples of type ' + data_type + ', epoch: ' + str(epoch))
    plt.show()


def train(G, D, data_type, seed_size, batch_size=50):
    num_epochs, d_steps, g_steps, = 8000, 10, 10
    criterion = nn.BCELoss()
    d_learning_rate, g_learning_rate, momentum = 0.0005, 0.0005, 0.9
    d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate)
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)
    g_scheduler = MultiStepLR(g_optimizer, milestones=[400, 1200, 4000], gamma=0.5)
    d_scheduler = MultiStepLR(d_optimizer, milestones=[400, 1200, 4000], gamma=0.5)

    one_label = torch.ones([batch_size, 1])
    zero_label = torch.zeros([batch_size, 1])
    print_freq = 100
    g = []
    df = []
    dr = []
    for e in range(num_epochs + 1):
        g_scheduler.step()
        d_scheduler.step()
        G.eval()
        for ds in range(d_steps):
            D.zero_grad()
            # train the discriminator on the real data: show a real example and update, label = 1
            # get a real example:
            real_data_set = Variable(get_data(batch_size, data_type))
            real_pred = D(real_data_set)
            real_error = criterion(real_pred, Variable(one_label))
            dr.append(real_error.item())
            real_error.backward()

            # train the discriminator on the fake data: let G create an example. label = 0
            seed = Variable(create_uniform_data(batch_size, seed_size))
            g_data = G(seed).detach()
            fake_pred = D(g_data)
            fake_error = criterion(fake_pred, Variable(zero_label))
            df.append(fake_error.item())
            fake_error.backward()

            d_optimizer.step()

        G.train()
        for gs in range(g_steps):
            # train G to fool D.
            G.zero_grad()
            seed = Variable(create_uniform_data(batch_size, seed_size))   # take a uniformly sampled vector
            g_data = G(seed)                        # pass it through G
            d_pred = D(g_data)                      # get the prediction of D on the output of G
            g_error = criterion(d_pred, Variable(one_label))  # the desired label is 1: meaning D think the example is real
            g.append(g_error.item())
            g_error.backward()
            g_optimizer.step()

        if e % print_freq == 0:
            print('{} d_r: {}\td_f: {}\tg: {}'.format(e, round(np.mean(dr), 3),
                                                      round(np.mean(df), 3), round(np.mean(g), 3)))
            dr.clear()
            df.clear()
            g.clear()
            plot_generated(G, seed_size, data_type, e)
            G.train()
    return G, D


def main(data_type):
    example_size = 2
    batch_size = 128
    d_input_size = example_size
    d_out = 1
    if data_type == 'line':
        seed_size = 2
    elif data_type == 'par':
        seed_size = 64
    else:
        seed_size = 8
    G = Generator(input_size=seed_size, output_size=example_size, drop_p=0.5)
    D = Discriminator(input_size=d_input_size, output_size=d_out)

    G, D = train(G, D, data_type, seed_size=seed_size, batch_size=batch_size)
    # test_output = G(create_uniform_data(n_points))
    # plot_data(test_output)


# data = get_data(1000, 'par')
# xs = data[:,0].tolist()
# ys = data[:,1].tolist()
# plt.plot(xs, ys, '.')
# plt.xlim([-1,1])
# plt.ylim([-1,1])
# plt.show()
main('spiral')
