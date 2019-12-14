import numpy as np
from scipy.special import factorial as fact

def coherentState(alpha, dim):
    n = np.arange(dim)
    state = np.exp(- 0.5 * np.abs(alpha) ** 2) / np.sqrt(fact([n])) * alpha ** n
    return state