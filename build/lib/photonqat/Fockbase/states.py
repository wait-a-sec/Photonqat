import numpy as np
from scipy.special import factorial as fact

def coherentState(alpha, dim):
    n = np.arange(dim)
    state = np.exp(- 0.5 * np.abs(alpha) ** 2) / np.sqrt(fact([n])) * alpha ** n
    return state

# arXiv:quant-ph/0509137
def catState(alpha, parity, dim):
    n = np.arange(dim)
    if parity == 'e':
        N = 1 / np.sqrt(2 * (1 + np.exp(-2 * np.abs(alpha) ** 2)))
        coeff = 2 * N * np.exp(-(np.abs(alpha) ** 2) / 2)
        state = coeff * alpha ** (n) / np.sqrt(fact(n)) * np.mod(n + 1, 2)
    elif parity == 'o':
        N = 1 / np.sqrt(2 * (1 - np.exp(-2 * np.abs(alpha)**2)))
        coeff = 2 * N * np.exp(-(np.abs(alpha) ** 2) / 2)
        state = coeff * alpha ** (n) / np.sqrt(fact(n)) * np.mod(n, 2)
    else:
        raise ValueError("parity must be 'e'(even) or 'o'(odd).")
    return state