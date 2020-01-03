
import numpy as np
from scipy.special import factorial as fact
import time

def FockWigner(xmat, pmat, fockState, mode, tol=1e-10):
    if fockState.ndim < mode + 1:
        raise  ValueError("The mode is not exist.")
    if fockState.ndim > 1:
        rho = reduceState(fockState, mode)
    else:
        rho = np.outer(np.conj(fockState), fockState)
    dim = len(fockState)
    grid = np.indices([dim, dim])
    W = FockWignerElement(xmat, pmat, grid[0], grid[1])
    W = rho * W
    W = np.sum(np.sum(W, axis = -1), axis = -1)
    if np.max(np.imag(W)) < tol:
        W = np.real(W)
    else:
        raise ValueError("Wigner plot has imaginary value.")
    return W

def reduceState(fockState, mode):
    modeNum = fockState.ndim
    cutoff = fockState.shape[-1] - 1
    fockState = np.swapaxes(fockState, mode, 0)
    fockState = fockState.flatten()
    rho = np.outer(np.conj(fockState), fockState)
    for i in range(modeNum - 1):
        rho = partialTrace(rho, cutoff)
    return  rho

def partialTrace(rho, cutoff):
    split = np.int(rho.shape[-1] / (cutoff + 1))
    rho = np.array(np.split(rho, split, axis = -1))
    rho = np.array(np.split(rho, split, axis = -2))
    rho = np.trace(rho, axis1 = -2, axis2 = -1)
    return  rho

def FockWignerElement(xmat, pmat, l, m):
    start = time.time()
    A = np.max(np.dstack([l, m]), axis=2)
    B = np.abs(l - m)
    C = np.min(np.dstack([l, m]), axis=2)
    R0 = xmat**2 + pmat**2
    xmat = xmat[:, :, np.newaxis, np.newaxis]
    pmat = pmat[:, :, np.newaxis, np.newaxis]
    R = xmat**2 + pmat**2
    X = xmat + np.sign(l-m) * 1j * pmat
    W = 2 * (-1)**C * np.sqrt(2**(B) * fact(C) / fact(A))
    W = W * np.exp(-R) * X**(B)
    S = Sonin(C, B, 2 * R0)
    return W * S

def to_2d_ndarray(a):
    if isinstance(a,(np.ndarray)):
        return a
    else:
        return np.array([[a]])

# slow!
def Sonin(n, alpha, x):
    start = time.time()
    n = to_2d_ndarray(n)
    alpha = to_2d_ndarray(alpha)
    x = to_2d_ndarray(x)
    a = fact(n + alpha)
    k0 = np.arange(np.max(n) + 1)
    k0 = k0[:, np.newaxis, np.newaxis]
    k = k0 * np.ones([np.max(n) + 1, n.shape[0], n.shape[0]], dtype = np.int)
    mask = np.ones(k.shape, dtype = np.int)
    for i in range(k.shape[0]):
        ind = (np.ones(n.shape) * i) > n
        mask[i, ind] = 0
    k *= mask
    S = mask * (-1)**k * a / fact(n - k) / fact(k + alpha) / fact(k)
    X = x ** k0
    S = S[:, np.newaxis, np.newaxis, :, :] * X[:, :, :, np.newaxis, np.newaxis]
    return np.sum(S, axis = 0)