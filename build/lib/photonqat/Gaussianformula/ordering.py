'''
References
arxiv: quant-ph/0503237
Physical Review A 49, 1567 (1994)

Conversion between vectors of deffrent ordered operators
R = (q_1, p_1, ..., q_n, p_n)
S = (q_1, ..., q_n, p_1, ..., p_n)
T = (a_1, ..., a^*_1, ..., a^*n)
'''

import numpy as np

def StoTmat(cov):
    N2 = cov.shape[0]
    N = np.int(N2/2)
    Omega = np.zeros([N2, N2]) + 0j
    for i in range(N):
        Omega[i, i] = 1 / np.sqrt(2)
        Omega[i, i + N] = 1j / np.sqrt(2)
        Omega[i + N, i] = 1 / np.sqrt(2)
        Omega[i + N, i + N] = -1j / np.sqrt(2)

    cov_T = np.dot(Omega, np.dot(cov, np.transpose(np.conj(Omega))))
    return cov_T

def RtoSmat(cov):
    N2 = cov.shape[0]
    N = np.int(N2/2)
    P = np.zeros([N2, N2]) + 0j
    for i in range(N2):
        if i%2:
            P[i, np.int(np.floor(i/2)) + N] = 1
        else:
            P[i, np.int(i/2)] = 1
    cov_T = np.dot(P, np.dot(cov, np.transpose(P)))
    return cov_T

def RtoTmat(cov):
    cov_T = RtoSmat(cov)
    return StoTmat(cov_T)

def StoTvec(vec):
    N2 = len(vec)
    N = np.int(N2/2)
    T = np.zeros(N2) + 0j
    T[:N] = (vec[:N] + 1j * vec[N:N2]) / np.sqrt(2)
    T[N:N2] = (vec[:N] - 1j * vec[N:N2]) / np.sqrt(2)
    return T

def RtoSvec(vec):
    N2 = len(vec)
    N = np.int(N2/2)
    S = np.zeros(N2) + 0j
    S[:N] = vec[0:N2:2]
    S[N:N2] = vec[1:N2:2]
    return S

def RtoTvec(vec):
    S = RtoSvec(vec)
    return StoTvec(S)
