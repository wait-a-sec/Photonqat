
import numpy as np
from scipy.special import factorial as fact
from scipy.linalg import expm

precision_factor = 2

def downMat(dim, order):    
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.zeros([dim, dim])
        for i in np.arange(order, dim):
            A[i, i - order] = np.prod(np.sqrt(np.arange(i, i - order, -1)))
        return A

def upMat(dim, order):        
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.zeros([dim, dim])
        for i in np.arange(0, dim - order):
            A[i, i + order] = np.prod(np.sqrt(np.arange(i + 1, i + 1 + order)))
        return A

def nMat(dim, order):
    if order == 0:
        A = np.eye(dim)
        return A
    else:
        A = np.diag(np.arange(dim) ** order)
        return A

def exp_annihiration(fockState, alpha, order = 1, cutoff = 10):
    row = fockState.shape[0]
    mat = downMat(fockState.shape[-1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    res = np.dot(fockState, mat_)
    return res

def exp_creation(fockState, alpha, order = 1, cutoff = 10):
    row = fockState.shape[0]
    mat = upMat(fockState.shape[-1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    res = np.dot(fockState, mat_)
    return res

def exp_photonNum(fockState, alpha, order = 1, cutoff = 10):
    row = fockState.shape[0]
    mat = nMat(fockState.shape[-1], order)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    res = np.dot(fockState, mat_)
    return res

def mat_for_mode2(mat):
    l = mat.shape[0]
    mat_ = np.zeros(np.array(mat.shape)**2)
    for i in range(mat.shape[0]):
        mat_[i*l:i*l+l, i*l:i*l+l] = mat
    return mat_

def mat_for_mode1(mat):
    l = mat.shape[0]
    mat_ = np.zeros(np.array(mat.shape)**2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            mat_[i*l:i*l+l, j*l:j*l+l] = np.eye(l) * mat[i, j]
    return mat_

def exp_BS(fockState, alpha, cutoff):
    state = np.zeros(fockState.shape) + 0j
    down = downMat(cutoff + 1, 1)
    up = upMat(cutoff + 1, 1)
    mat1_ = np.dot(mat_for_mode1(up), mat_for_mode2(down))
    mat2_ = np.dot(mat_for_mode1(down), mat_for_mode2(up))
    mat_ = mat1_ - mat2_
    emat_ = expm(alpha * mat_)
    res = np.dot(fockState, emat_)
    return res

def exp_AAaa(fockState, alpha, cutoff):
    mat = downMat(fockState.shape[-1], 2)
    mat = np.dot(upMat(fockState.shape[-1], 2), mat)
    mat_ = np.empty(mat.shape, dtype=np.complex)
    mat_ = expm(alpha * mat)
    res = np.dot(fockState, mat_)
    return res