import numpy as np
import matplotlib.pyplot as plt
from .Fockbase.gates import *
from .Fockbase.states import *
from .Fockbase.WignerFunc import *

STATE_SET = {
    "vacuum": vacuumState,
    "coherent": coherentState,
    "cat": catState,
    "n_photon": photonNumberState
}

GATE_SET = {
    "D": Dgate,
    "BS": BSgate,
    "S": Sgate,    
    "Kerr": Kgate,
    "MeasF": MeasF
}

class Fock():
    def __init__(self, N, cutoff = 10):
        self.N = N
        self.cutoff = cutoff
        #self.dim = cutoff + 1
        self.initState = np.zeros([N, self.cutoff + 1]) + 0j
        self.initState[:, 0] = 1
        self.state = None
        self.initStateFlag = False
        self.ops = []
        self.creg = [[None, None, None] for i in range(self.N)] # [x, p, n]

    def  __getattr__(self, name):
        if name in STATE_SET:
            if self.initStateFlag:
                raise ValueError("State must be set before gate operation.")
            self.ops.append(STATE_SET[name])
            return self._setGateParam

        elif name in GATE_SET:
            self.ops.append(GATE_SET[name])
            self.initStateFlag = True
            return self._setGateParam

        else:
            raise AttributeError('The method does not exist')

    def _setGateParam(self, *args, **kwargs):
        self.ops[-1] = self.ops[-1](self, *args, **kwargs)
        return self

    def Creg(self, idx, var, scale = 1):
        return CregReader(self.creg, idx, var, scale)

    def run(self):
        for op in self.ops:
            if isinstance(op, STATE):
                self.initState = op.run(self.initState)
            elif isinstance(op, GATE):
                if self.state is None:
                    self.state = self._multiTensordot()
                self.state = op.run(self.state)
        return self

    def _multiTensordot(self):
        self.state = self.initState[0, :]
        for i in range(self.N - 1):
            self.state = np.tensordot(self.state, self.initState[i+1, :], axes = 0)
        return self.state

    def Wigner(self, mode, method = 'clenshaw', plot = 'y', xrange = 5.0, prange = 5.0):
        if self.state is None:
            self.state = self._multiTensordot()
            self.initState == None
        x = np.arange(-xrange, xrange, xrange / 50)
        p = np.arange(-prange, prange, prange / 50)
        m = len(x)
        xx, pp = np.meshgrid(x, -p)
        W = FockWigner(xx, pp, self.state, mode, method)
        if plot == 'y':
            h = plt.contourf(x, p, W)
            plt.show()
        return (x, p, W)

    def photonSampling(self, mode, ite = 1):
        if self.state is None:
            self.state = self._multiTensordot()
            self.initState == None
        reducedDensity = reduceState(self.state, mode)
        probs = np.real(np.diag(reducedDensity))
        probs = probs / np.sum(probs)
        return np.random.choice(probs.shape[0], ite, p = probs)
