import numpy as np
from .stateOps import *

class STATE():
    def __init__(self, obj):
        self.obj = obj

class coherentState(STATE):
    def __init__(self, obj, mode, alpha):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.mode = mode
        self.alpha = alpha
        super().__init__(obj)

    def run(self, initState):
        return coherent(initState, self.mode, self.alpha, self.cutoff)

class vacuumState(STATE):
    def __init__(self, obj, mode):
        self.obj = obj
        self.mode = mode
        super().__init__(obj)

    def run(self, initState):
        return vacuum(initState, self.mode)

class catState(STATE):
    def __init__(self, obj, mode, alpha, parity):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.mode = mode
        self.alpha = alpha
        self.parity = parity
        if (parity != "e") and (parity != "o"):
            raise ValueError("parity must be 'e'(even) or 'o'(odd).")
        
    def run(self, initState):
        return cat(initState, self.mode, self.alpha, self.parity, self.cutoff)

class photonNumberState(STATE):
    def __init__(self, obj, mode, n):
        self.obj = obj
        self.cutoff = self.obj.cutoff
        self.mode = mode
        self.n = n
    def run(self, initState):
        return photonNumState(initState, self.mode, self.n, self.cutoff)
