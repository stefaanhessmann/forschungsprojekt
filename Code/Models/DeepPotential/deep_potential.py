from torch import nn


class DeepPotential(nn.Module):

    def __init__(self):
        sub_dim = 18 * 4
        super(DeepPotential, self).__init__()
        for subnetwork in ['h_net', 'o_net', 'c_net']:
            setattr(self, subnetwork, nn.Sequential(nn.Linear(sub_dim, 600),
                                                    nn.ReLU(),
                                                    # nn.BatchNorm1d(600),
                                                    nn.Linear(600, 400),
                                                    nn.ReLU(),
                                                    # nn.BatchNorm1d(400),
                                                    nn.Linear(400, 200),
                                                    nn.ReLU(),
                                                    # nn.BatchNorm1d(200),
                                                    nn.Linear(200, 100),
                                                    nn.ReLU(),
                                                    # nn.BatchNorm1d(100),
                                                    nn.Linear(100, 80),
                                                    nn.ReLU(),
                                                    # nn.BatchNorm1d(80),
                                                    nn.Linear(80, 40),
                                                    nn.ReLU(),
                                                    # nn.BatchNorm1d(40),
                                                    nn.Linear(40, 20),
                                                    nn.ReLU(),
                                                    # nn.BatchNorm1d(20),
                                                    nn.Linear(20, 10),
                                                    nn.ReLU(),
                                                    # nn.BatchNorm1d(10),
                                                    nn.Linear(10, 1),
                                                    nn.ReLU()))

    def forward(self, X):
        a1 = self.h_net(X[:, 0])
        a2 = self.h_net(X[:, 1])
        a3 = self.h_net(X[:, 2])
        a4 = self.h_net(X[:, 3])
        a5 = self.h_net(X[:, 4])
        a6 = self.h_net(X[:, 5])
        a7 = self.h_net(X[:, 6])
        a8 = self.h_net(X[:, 7])
        a9 = self.h_net(X[:, 8])
        a10 = self.h_net(X[:, 9])
        a11 = self.c_net(X[:, 10])
        a12 = self.c_net(X[:, 11])
        a13 = self.c_net(X[:, 12])
        a14 = self.c_net(X[:, 13])
        a15 = self.c_net(X[:, 14])
        a16 = self.c_net(X[:, 15])
        a17 = self.c_net(X[:, 16])
        a18 = self.o_net(X[:, 17])
        a19 = self.o_net(X[:, 18])
        out = a1 + a2 + a3 + a4 + a5 + a6 \
              + a7 + a8 + a9 + a10 + a11 \
              + a12 + a13 + a14 + a15 + a16 \
              + a17 + a18 + a19
        return out