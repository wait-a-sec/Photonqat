
import numpy as np
from .bosonicLadder import *


def displacement(fockState, mode, alpha, cutoff = 10):
    modeNum = fockState.ndim
    state = singleGate_preProcess(fockState, mode)
    state = exp_annihiration(state, -np.conj(alpha), cutoff = cutoff)
    state = exp_creation(state, alpha, cutoff = cutoff)
    state = singleGate_postProcess(state, mode, modeNum)
    return state * np.exp(-np.abs(alpha)**2 / 2)

def BS(fockState, mode1, mode2, theta, cutoff):
    modeNum = fockState.ndim
    if modeNum < 2:
        raise ValueError("The gate requires more than one mode.")
    state = twoModeGate_preProcess(fockState, mode1, mode2)
    state = exp_BS(state, -theta, cutoff)
    state = twoModeGate_postProcess(state, mode1, mode2, modeNum)
    return state

def squeeze(fockState, mode, r, phi, cutoff):
    G = np.exp(2 * 1j * phi) * np.tanh(r)
    modeNum = fockState.ndim
    state = singleGate_preProcess(fockState, mode)
    state = exp_annihiration(state, np.conj(G) / 2, order = 2, cutoff = cutoff)
    state = exp_photonNum(state, -np.log(np.cosh(r)), cutoff = cutoff)
    state = exp_creation(state, -G / 2, order = 2, cutoff = cutoff)
    state = singleGate_postProcess(state, mode, modeNum)
    return state / np.sqrt(np.cosh(r))

def singleGate_preProcess(fockState, mode):
    cutoff = fockState.shape[-1] - 1
    fockState = np.swapaxes(fockState, mode, fockState.ndim - 1)
    return fockState.reshape(-1, cutoff + 1)

def twoModeGate_preProcess(fockState, mode1, mode2):
    cutoff = fockState.shape[-1] - 1
    modeNum = fockState.ndim
    fockState = np.swapaxes(fockState, mode2, modeNum - 1)
    fockState = np.swapaxes(fockState, mode1, modeNum - 2)
    return fockState.reshape(-1, cutoff + 1, cutoff + 1)

def singleGate_postProcess(fockState, mode, modeNum):
    cutoff = fockState.shape[-1] - 1
    fockState = fockState.reshape([cutoff + 1] * modeNum)
    return np.swapaxes(fockState, mode, modeNum - 1)

def twoModeGate_postProcess(fockState, mode1, mode2, modeNum):
    cutoff = fockState.shape[-1] - 1
    fockState = fockState.reshape([cutoff + 1] * modeNum)
    fockState = np.swapaxes(fockState, mode1, modeNum - 2)
    fockState = np.swapaxes(fockState, mode2, modeNum - 1)
    return fockState