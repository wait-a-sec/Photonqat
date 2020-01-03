import numpy as np
import matplotlib.pyplot as plt
from .Fockbase.gates import *
from .Fockbase.states import *
from .Fockbase.WignerFunc import *

class Fock():
    def __init__(self, N, cutoff = 10):
        self.N = N
        self.cutoff = cutoff
        self.dim = cutoff + 1
        self.initState = np.zeros([N, self.dim]) + 0j
        self.initState[:, 0] = 1
        self.state = None

    def vacuumState(self, mode):
        #fockState = np.zeros([self.dim ** self.N])
        self.initState[mode, :] = 0
        self.initState[mode, 0] = 1

    def multiTensordot(self):
        self.state = self.initState[0, :]
        for i in range(self.N - 1):
            self.state = np.tensordot(self.state, self.initState[i+1, :], axes = 0)
        return self.state

    def Wignerfunc(self, mode, plot = 'y', xrange = 5.0, prange = 5.0):
        if self.state is None:
            self.state = self.multiTensordot()
            self.initState == None
        x = np.arange(-xrange, xrange, xrange / 50)
        p = np.arange(-prange, prange, prange / 50)
        m = len(x)
        xx, pp = np.meshgrid(x, p)
        xi_array = np.dstack((pp, xx))
        W = FockWigner(xx, pp, self.state, mode)
        if plot == 'y':
            h = plt.contourf(x, p, W)
            plt.show()
        return (x, p, W)
    
    def coherentState(self, mode, alpha):
        if self.initState is None:
            raise ValueError("State must be set before gate operation.")
        self.initState[mode, :] = coherentState(alpha, self.dim)
    
    def catState(self, mode, alpha, parity):
        if self.initState is None:
            raise ValueError("State must be set before gate operation.")
        self.initState[mode, :] = catState(alpha, parity, self.dim)

    def photonNumberState(self, mode, N):
        if self.initState is None:
            raise ValueError("State must be set before gate operation.")
        photonNumState = np.zeros(self.dim)
        photonNumState[N] = 1
        self.initState[mode, :] = photonNumState

    def Dgate(self, mode, alpha):
        if self.state is None:
            self.state = self.multiTensordot()
            self.initState == None
        self.state = displacement(self.state, mode, alpha, self.cutoff)

    def BSgate(self, mode1, mode2, theta = np.pi/4):
        if self.state is None:
            self.state = self.multiTensordot()
            self.initState == None
        self.state = BS(self.state, mode1, mode2, theta, self.cutoff)

    def Sgate(self, mode, r, phi = 0):
        if self.state is None:
            self.state = self.multiTensordot()
            self.initState == None
        self.state = squeeze(self.state, mode, r, phi, self.cutoff)

    def Kgate(self, mode, chi):
        if self.state is None:
            self.state = self.multiTensordot()
            self.initState == None
        self.state = kerr(self.state, mode, chi, self.cutoff)

    def photonSampling(self, mode, ite = 1):
        reducedDensity = reduceState(self.state, mode)
        probs = np.real(np.diag(reducedDensity))
        probs = probs / np.sum(probs)
        return np.random.choice(probs.shape[0], ite, p = probs)

    def photonMeasurement(self, mode, post_select = None):
        reducedDensity = reduceState(self.state, mode)
        probs = np.real(np.diag(reducedDensity))
        probs = probs / np.sum(probs)
        if post_select is None:
            res = np.random.choice(probs.shape[0], 1, p = probs)
        else:
            res = post_select
        prob = probs[res]
        
        state_ = np.swapaxes(self.state, mode, 0)
        ind = np.ones((self.state.shape[-1]), bool)
        ind[res] = False
        state_[ind] = 0
        state_ = np.swapaxes(state_, mode, 0)
        self.state = state_ / np.sqrt(prob)
        return res