import numpy as np
from .baseFunc import *
import matplotlib.pyplot as plt

class Gaussian_trans():
    def __init__(self, N):
        self.N = N
        self.V = np.eye(2 * N)
        self.mu = np.zeros(2 * N)

    def mean(self, idx):
        res = np.copy(self.mu[2 * idx:2 * idx + 2])
        return res

    def cov(self, idx):
        res = np.copy(self.V[(2 * idx):(2 * idx + 2), (2 * idx):(2 * idx + 2)])
        return res

    def Xsqueeze(self, idx, r):
        idx = 2 * idx
        S = np.eye(2 * self.N)
        S[idx:idx+2, idx:idx+2] = np.array([[np.exp(-r), 0], [0, np.exp(r)]])
        self.V = np.dot(S, np.dot(self.V, S.T))
        self.mu = np.dot(S, self.mu)
        
    def Psqueeze(self, idx, r):
        idx = 2 * idx
        S = np.eye(2 * self.N)
        S[idx:idx+2, idx:idx+2] = np.array([[np.exp(r), 0], [0, np.exp(-r)]])
        self.V = np.dot(S, np.dot(self.V, S.T))
        self.mu = np.dot(S, self.mu)
        
    def rotation(self, idx, theta):
        idx = 2 * idx
        S = np.eye(2 * self.N)
        S[idx:idx+2, idx:idx+2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.V = np.dot(S, np.dot(self.V, S.T))
        self.mu = np.dot(S, self.mu)
        
    def BS(self, idx1, idx2, theta):
        idx1 = 2 * idx1
        idx2 = 2 * idx2
        S = np.eye(2 * self.N)
        S[idx1:idx1+2, idx1:idx1+2] = np.array([[np.sin(theta), 0], [0, np.sin(theta)]])
        S[idx1:idx1+2, idx2:idx2+2] = np.array([[np.cos(theta), 0], [0, np.cos(theta)]])
        S[idx2:idx2+2, idx1:idx1+2] = np.array([[np.cos(theta), 0], [0, np.cos(theta)]])
        S[idx2:idx2+2, idx2:idx2+2] = np.array([[-np.sin(theta), 0], [0, -np.sin(theta)]])
        self.V = np.dot(S, np.dot(self.V, S.T))
        self.mu = np.dot(S, self.mu)
        
    def twoModeSqueezing(self, idx1, idx2,  r):
        idx1 = 2 * idx1
        idx2 = 2 * idx2
        S = np.eye(2 * self.N)
        S[idx1:idx1+2, idx1:idx1+2] = np.array([[np.cosh(r), 0], [0, np.cosh(r)]])
        S[idx1:idx1+2, idx2:idx2+2] = np.array([[np.sinh(r), 0], [0, -np.sinh(r)]])
        S[idx2:idx2+2, idx1:idx1+2] = np.array([[np.sinh(r), 0], [0, -np.sinh(r)]])
        S[idx2:idx2+2, idx2:idx2+2] = np.array([[np.cosh(r), 0], [0, np.cosh(r)]])
        self.V = np.dot(S, np.dot(self.V, S.T))
        self.mu = np.dot(S, self.mu)        
    
    def Displace(self, idx, alpha):
        dx = 2 * np.real(alpha)
        dp = 2 * np.imag(alpha)
        self.mu[2 * idx:2 * idx + 2] = self.mu[2 * idx:2 * idx + 2] + np.array([dx, dp])

    def Xgate(self, idx, dx):
        self.mu[2 * idx] += dx

    def Zgate(self, idx, dp):
        self.mu[2 * idx + 1] += dp
        
    def MeasureX(self, idx):
        res = np.random.normal(self.mu[2 * idx], np.sqrt(self.V[2 * idx, 2 * idx]))
        self.mu, self.V = StateAfterMeasurement(self.mu, self.V, idx, res, np.diag([1, 0]))        
        return res
    
    def MeasureP(self, idx):
        res = np.random.normal(self.mu[2 * idx + 1], np.sqrt(self.V[2 * idx + 1, 2 * idx + 1]))
        self.mu, self.V = StateAfterMeasurement(self.mu, self.V, idx, res, np.diag([0, 1]))
        return res

    def plotGaussianWigner(self, idx):
        idx = idx * 2
        x = np.arange(-5, 5, 0.1)
        p = np.arange(-5, 5, 0.1)
        m = len(x)
        xx, pp = np.meshgrid(x, p)
        xi_array = np.dstack((pp, xx))
        W = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                W[i][j] = GaussianWigner(xi_array[j][i], self.V[idx:idx+2, idx:idx+2], self.mu[idx:idx+2])
        h = plt.contourf(x, p, W)
        plt.show()