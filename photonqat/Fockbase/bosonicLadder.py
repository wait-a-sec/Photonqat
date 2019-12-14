
import numpy as np
from scipy.special import factorial as fact

precision_factor = 2

def down(order, n):
    n_ = n - order
    n_ = np.clip(n_, 0, None)
    coeff = np.sqrt(fact(n) / fact(n_))
    coeff[(n - order) < 0] = 0
    return [coeff, n_.astype(np.int)]

def up(order, n, cutoff = 10):
    n_ = n + order
    n_[n_ > cutoff] = 0
    coeff = np.sqrt(fact(n_) / fact(n))
    coeff[(n + order) > cutoff] = 0
    return [coeff, n_.astype(np.int)]

def photonNum(order, n, cutoff = 10):
    coeff = n ** order
    return [coeff, n.astype(np.int)]

def exp_annihiration(fockState, alpha = 1, order = 1, cutoff = 10):
    ind = np.arange(fockState.shape[-1])
    state = np.zeros(fockState.shape) + 0j
    for i in range(precision_factor * (cutoff + 1)):
        tmp = down(order * i, ind)
        state[:, tmp[1]] += tmp[0] * fockState[:, ind] / fact([i]) * alpha ** i
    return state

def exp_creation(fockState, alpha = 1, order = 1, cutoff = 10):
    ind = np.arange(fockState.shape[-1])
    state = np.zeros(fockState.shape) + 0j
    for i in range(precision_factor * (cutoff + 1)):
        tmp =  up(order * i, ind, cutoff = cutoff)
        state[:, tmp[1]] += tmp[0] * fockState[:, ind] / fact([i]) * alpha ** i
    return state

def exp_photonNum(fockState, alpha, order = 1, cutoff = 10):
    ind = np.arange(fockState.shape[-1])
    state = np.zeros(fockState.shape) + 0j
    for i in range(precision_factor * (cutoff + 1)):
        tmp =  photonNum(order * i, ind, cutoff)  # (order, n)
        state[:, tmp[1]] += tmp[0] * fockState[:, ind] / fact([i]) * alpha ** i
    return state

def Ab_Ba(fockState, cutoff):
    state = np.zeros(fockState.shape) + 0j
    dim = cutoff + 1
    a = np.arange(dim) * np.ones([dim, dim])
    ind1 = np.ravel(a, order = 'F').astype(np.int)
    ind2 = np.ravel(a, order = 'K').astype(np.int)
    coef =  fockState[:, ind1, ind2]
    down1 = down(1, ind1)
    up2 = up(1, ind2, cutoff = cutoff)
    state[:, down1[1], up2[1]] -= down1[0] * up2[0] * coef

    up1 = up(1, ind1, cutoff = cutoff)
    down2 = down(1, ind2)
    state[:, up1[1], down2[1]] += up1[0] * down2[0] * coef
    return state

def pow_Ab_Ba(fockState, n, cutoff = 10):
    state = np.copy(fockState)
    if n == 0:
        return state
    else:
        for i in range(n):
            state = Ab_Ba(state, cutoff = cutoff)
        return state

def exp_BS(fockState, alpha, cutoff = 10):
    state = np.zeros(fockState.shape) + 0j
    for i in range(precision_factor * (cutoff + 1)): # order
        tmpstate = np.copy(fockState)
        tmpstate = pow_Ab_Ba(tmpstate, i, cutoff = cutoff)
        tmpstate = tmpstate / fact([i]) * alpha ** i
        state += tmpstate
    return state