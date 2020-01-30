
"""
`gateOps` module implements calculation for quantum gate operations.
This module may be redefined as a backend option in later versions.
"""


import numpy as np
from .bosonicLadder import *
from .WignerFunc import reduceState

def Displacement(state, mode, alpha, modeNum, cutoff):
    state = _singleGate_preProcess(state, mode)
    state = exp_annihiration(state, -np.conj(alpha), cutoff = cutoff)
    state = exp_creation(state, alpha, cutoff = cutoff)
    state = _singleGate_postProcess(state, mode, modeNum)
    state = state * np.exp(-np.abs(alpha)**2 / 2)
    return state

def Beamsplitter(state, mode1, mode2, theta, modeNum, cutoff):
    if modeNum < 2:
        raise ValueError("The gate requires more than one mode.")
    state = _twoModeGate_preProcess(state, mode1, mode2)
    state = exp_BS(state, -theta, cutoff = cutoff)
    state = _twoModeGate_postProcess(state, mode1, mode2, modeNum)
    return state

def Squeeze(state, mode, r, phi, modeNum, cutoff):
    G = np.exp(2 * 1j * phi) * np.tanh(r)
    state = _singleGate_preProcess(state, mode)
    state = exp_annihiration(state, np.conj(G) / 2, order = 2, cutoff = cutoff)
    state = exp_photonNum(state, -np.log(np.cosh(r)), cutoff = cutoff)
    state = exp_creation(state, -G / 2, order = 2, cutoff = cutoff)
    state = _singleGate_postProcess(state, mode, modeNum)
    state = state / np.sqrt(np.cosh(r))
    return state

def KerrEffect(state, mode, chi, modeNum, cutoff):
    state = _singleGate_preProcess(state, mode)
    state = exp_AAaa(state, 1j * chi / 2, cutoff = cutoff)
    state = _singleGate_postProcess(state, mode, modeNum)
    return state

def photonMeasurement(state, mode, post_select):
    reducedDensity = reduceState(state, mode)
    probs = np.real(np.diag(reducedDensity))
    probs = probs / np.sum(probs)
    if post_select is None:
        res = np.random.choice(probs.shape[0], 1, p = probs)
    else:
        res = post_select
    prob = probs[res]
    
    state_ = np.swapaxes(state, mode, 0)
    ind = np.ones((state.shape[-1]), bool)
    ind[res] = False
    state_[ind] = 0
    state_ = np.swapaxes(state_, mode, 0)
    state_ = state_ / np.sqrt(prob)
    return res, state_

def _singleGate_preProcess(fockState, mode):
    cutoff = fockState.shape[-1] - 1
    fockState = np.swapaxes(fockState, mode, fockState.ndim - 1)
    return fockState.reshape(-1, cutoff + 1)

def _singleGate_postProcess(fockState, mode, modeNum):
    cutoff = fockState.shape[-1] - 1
    fockState = fockState.reshape([cutoff + 1] * modeNum)
    return np.swapaxes(fockState, mode, modeNum - 1)

def _twoModeGate_preProcess(fockState, mode1, mode2):
    cutoff = fockState.shape[-1] - 1
    modeNum = fockState.ndim
    fockState = np.swapaxes(fockState, mode2, modeNum - 1)
    fockState = np.swapaxes(fockState, mode1, modeNum - 2)
    return fockState.reshape(-1, (cutoff + 1) ** 2)

def _twoModeGate_postProcess(fockState, mode1, mode2, modeNum):
    dim = np.int(np.sqrt(fockState.shape[-1]))
    fockState = fockState.reshape([dim] * modeNum)
    fockState = np.swapaxes(fockState, mode1, modeNum - 2)
    fockState = np.swapaxes(fockState, mode2, modeNum - 1)
    return fockState
