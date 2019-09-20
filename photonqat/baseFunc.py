import numpy as np

def GaussianWigner(xi, V, mu):
    xi = xi - mu
    xi_tmp = np.ravel(xi)
    N = np.int(len(xi_tmp) / 2)
    det_V = np.linalg.det(V)
    V_inv = np.linalg.inv(V)
    W = (2 * np.pi)**(-N) / np.sqrt(det_V) * np.exp(-1/2 * np.dot(xi_tmp, np.dot(V_inv, xi_tmp.T)))
    return W

def StateAfterMeasurement(mu, V, idx, res, Pi):
    N = np.int(V.shape[0] / 2)
    subSysA = np.delete(np.delete(V, [2 * idx, 2 * idx + 1], 0), [2 * idx, 2 * idx + 1], 1)
    subSysB = V[(2 * idx):(2 * idx + 2), (2 * idx):(2 * idx + 2)]
    arrayList = []
    for j in range(N):
        if j != idx:
            arrayList.append(V[(2 * j):(2 * j + 2), (2 * idx):(2*idx + 2)])
    C = np.concatenate(arrayList)
    post_V = subSysA - np.dot(C, np.dot(1 / np.sum(subSysB * Pi) * Pi, C.T))
    post_V = np.insert(post_V, 2 * idx, [[0], [0]], axis = 0)
    post_V = np.insert(post_V, 2 * idx, [[0], [0]], axis = 1)
    post_V[2 * idx, 2 * idx] = 1
    post_V[2 * idx + 1, 2 * idx + 1] = 1
    
    post_mu = np.delete(mu, [2 * idx, 2 * idx + 1]) - \
    np.dot(np.dot(C, 1 / np.sum(subSysB * Pi) * Pi), res * np.diag(Pi) - mu[(2 * idx):(2 * idx + 2)])
    post_mu = np.insert(post_mu, 2 * idx, [0, 0])
    
    return post_mu, post_V