import numpy as np
from .Gaussianformula.baseFunc import *
from .Gaussianformula.ordering import *
import matplotlib.pyplot as plt

class Gaussian():
    def __init__(self, N):
        self.N = N
        self.V = (np.eye(2 * N)) * 0.5
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
        dx = np.real(alpha) * np.sqrt(2) # np.sqrt(2 * hbar)
        dp = np.imag(alpha) * np.sqrt(2) # np.sqrt(2 * hbar)
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


    def Wignerfunc(self, idx, plot = 'y', xrange = 5.0, prange = 5.0):
        idx = idx * 2
        x = np.arange(-xrange, xrange, 0.1)
        p = np.arange(-prange, prange, 0.1)
        m = len(x)
        xx, pp = np.meshgrid(x, p)
        xi_array = np.dstack((pp, xx))
        W = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                W[i][j] = GaussianWigner(xi_array[j][i], self.V[idx:idx+2, idx:idx+2], self.mu[idx:idx+2])
        if plot == 'y':
            h = plt.contourf(x, p, W)
            plt.show()
        return (x, p, W)

        
    def PhotonDetectionProb(self, m, n):
        if len(m) != self.N or len(n) != self.N:
            raise ValueError("Input array dimension must be same as mode Number.")
        return np.real(FockDensityMatrix(self.V, self.mu, m, n))


    def GaussianToFock(self, cutoffDim = 10):
        photonNumList = []
        cutoffDim += 1
        rho = np.empty([cutoffDim ** self.N, cutoffDim ** self.N])
        for i in range(cutoffDim ** self.N):
            photonNum = []
            for j in range(self.N):
                photonNum.insert(0, np.int(np.floor(i / (cutoffDim ** j))) % cutoffDim)
            photonNumList.append(photonNum)

        for i in range(cutoffDim ** self.N):
            for j in range(cutoffDim ** self.N):
                m = np.array(photonNumList[i])
                n = np.array(photonNumList[j])
                row = [m[i] ** (self.N - i - 1) for i in range(self.N)]
                col = [n[i] ** (self.N - i - 1) for i in range(self.N)]
                rho[row, col] = FockDensityMatrix(self.V, self.mu, m, n)

        return rho
