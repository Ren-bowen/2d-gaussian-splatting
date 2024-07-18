import numpy as np
import cupy as cp
from phys_engine import utils
import scipy.sparse as sparse
import multiprocessing
import time

def cupy_block_2d(matrices):
    assert len(matrices) == 2 and len(matrices[0]) == 2
    top = cp.concatenate(matrices[0], axis=1)
    bottom = cp.concatenate(matrices[1], axis=1)
    return cp.concatenate([top, bottom], axis=0)

def val(x, e, l2, k):
    sum = 0.0
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        sum += l2[i] * 0.5 * k[i] * (diff.dot(diff) / l2[i] - 1) ** 2
    return sum

def grad(x, e, l2, k):
    g = np.array([[0.0, 0.0, 0.0]] * len(x))
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        g_diff = 2 * k[i] * (diff.dot(diff) / l2[i] - 1) * diff
        g[e[i][0]] += g_diff
        g[e[i][1]] -= g_diff
    return g

def hess(x, e, l2, k):
    IJV = [cp.zeros(len(e) * 36, dtype=cp.int32), cp.zeros(len(e) * 36, dtype=cp.int32), cp.zeros(len(e) * 36, dtype=cp.float32)]

    for i in range(0, len(e)):
        diff = cp.asarray(x[e[i][0]]) - cp.asarray(x[e[i][1]])
        H_diff = 2 * k[i] / l2[i] * (2 * cp.outer(diff, diff) + (diff.dot(diff) - l2[i]) * cp.identity(3))
        H_local = utils.make_PSD(cupy_block_2d([[H_diff, -H_diff], [-H_diff, H_diff]]))  
        # add to global matrix
        for nI in range(0, 2):
            for nJ in range(0, 2):
                indStart = i * 36 + (nI * 2 + nJ) * 9
                for r in range(0, 3):
                    for c in range(0, 3):
                        IJV[0][indStart + r * 3 + c] = e[i][nI] * 3 + r
                        IJV[1][indStart + r * 3 + c] = e[i][nJ] * 3 + c
                        IJV[2][indStart + r * 3 + c] = H_local[nI * 3 + r, nJ * 3 + c]

    return [cp.asnumpy(IJV[0]), cp.asnumpy(IJV[1]), cp.asnumpy(IJV[2])]