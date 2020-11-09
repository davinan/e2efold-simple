import torch.nn as nn
import torch


class PostProcessingNetwork(nn.Module):
    def __init__(self, n_iterations):
        super(PostProcessingNetwork, self).__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.s = nn.Parameter(torch.randn(1))
        self.k = nn.Parameter(torch.randn(1))
        self.T = n_iterations
        self.pp_cell = PPcell()
        self.relu = nn.ReLU()

    def forward(self, U, M):
        tau = lambda a: 0.5*(a*a+(a*a).transpose(1, 2))*M
        ones = torch.ones(U.size()[1:2])
        U = torch.sigmoid((U - self.s) * self.k) * U
        A_hat = torch.sigmoid((U - self.s)*self.k) * torch.sigmoid(U)
        print(A_hat.size())
        print(M.shape)
        A = tau(A_hat)
        print(A.shape, ones.shape)

        lambd = self.w * self.relu(A.sum(dim=1) - ones)#self.relu(A.dot(ones) - ones)

        for _ in range(self.T):
            lambd, A, A_hat = self.pp_cell(U, M, A, A_hat, lambd)

        return A
#
# class Tau(nn.Module):
#     def __init__(self):
#         super(Tau, self).__init__()
#
#     def forward(self, a,  M):
#         return 0.5*(a*a+(a*a).transpose(1, 2))*M

class PPcell(nn.Module):
    def __init__(self):
        super(PPcell, self).__init__()
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))
        self.y_a = nn.Parameter(torch.randn(1))
        self.y_b = nn.Parameter(torch.randn(1))
        self.p = nn.Parameter(torch.randn(1))
        self.relu = nn.ReLU()
        self.k = nn.Parameter(torch.randn(1))

    def forward(self, U, M, A, A_hat, lambd):
        tau = lambda a: 0.5*(a*a+(a*a).transpose(1, 2))*M
        ones = torch.ones(A.size()[:2])
        g = 0.5 * U - (lambd * torch.sigmoid((A.sum(dim=1) - ones) * self.k)).matmul(ones.transpose(1, 0))
        A_dot = A_hat + self.alpha * self.y_a * A_hat * M * (g + g.transpose(1, 2))
        A_hat = self.relu(torch.abs(A_dot) - self.p*self.alpha*self.y_a)
        A_hat = 1 - self.relu(1 - A_hat)
        A = tau(A_hat)
        lambd += self.beta*self.y_b*self.relu(A.sum(dim=1) - 1)
        print(A.sum())
        return lambd, A, A_hat