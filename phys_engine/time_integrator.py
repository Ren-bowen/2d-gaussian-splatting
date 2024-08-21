import copy
from cmath import inf

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
import time
from scipy.sparse.linalg import spsolve
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import cg

from phys_engine import InertiaEnergy
from phys_engine import MassSpringEnergy

def step_forward(x, e, v, m, l2, k, h, tol):
    x_tilde = x + v * h     # implicit Euler predictive position
    x_n = copy.deepcopy(x)

    # Newton loop
    iter = 0
    E_last = IP_val(x, e, x_tilde, m, l2, k, h)
    p = search_dir(x, e, x_tilde, m, l2, k, h)
    while LA.norm(p, inf) / h > tol:
        print('Iteration', iter, ':')
        print('residual =', LA.norm(p, inf) / h)

        # line search
        alpha = 1
        while IP_val(x + alpha * p, e, x_tilde, m, l2, k, h) > E_last:
            alpha /= 2
        print('step size =', alpha)

        x += alpha * p
        E_last = IP_val(x, e, x_tilde, m, l2, k, h)
        t0 = time.time()
        p = search_dir(x, e, x_tilde, m, l2, k, h)
        print('search_dir time:', time.time() - t0)
        iter += 1

    v = (x - x_n) / h   # implicit Euler velocity update
    return [x, v]

def IP_val(x, e, x_tilde, m, l2, k, h):
    return InertiaEnergy.val(x, x_tilde, m) + h * h * MassSpringEnergy.val(x, e, l2, k)     # implicit Euler

def IP_grad(x, e, x_tilde, m, l2, k, h):
    return InertiaEnergy.grad(x, x_tilde, m) + h * h * MassSpringEnergy.grad(x, e, l2, k)   # implicit Euler

def IP_hess(x, e, x_tilde, m, l2, k, h):
    start_time = time.time()
    IJV_In = InertiaEnergy.hess(x, x_tilde, m)
    print('Inertia time:', time.time() - start_time)
    start_time = time.time()
    IJV_MS = MassSpringEnergy.hess(x, e, l2, k)
    print('MassSpring time:', time.time() - start_time)
    IJV_MS[2] *= h * h    # implicit Euler
    IJV = np.append(IJV_In, IJV_MS, axis=1)
    H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 3, len(x) * 3)).tocsr()
    return H

def search_dir(x, e, x_tilde, m, l2, k, h):
    start_time = time.time()
    projected_hess = IP_hess(x, e, x_tilde, m, l2, k, h)
    start_time = time.time()
    reshaped_grad = IP_grad(x, e, x_tilde, m, l2, k, h).reshape(len(x) * 3, 1)
    print('grad_time: ', time.time() - start_time)
    projected_hess = csr_matrix((cp.asarray(projected_hess.data),
                                cp.asarray(projected_hess.indices),
                                cp.asarray(projected_hess.indptr)),
                                shape=projected_hess.shape)
    start_time = time.time()
    reshaped_grad = cp.asarray(reshaped_grad)
    print('reshaped_grad_time:', time.time() - start_time)
    start_time = time.time()
    solution, info = cg(projected_hess, -reshaped_grad)
    print('CG time:', time.time() - start_time)
    return cp.asnumpy(solution.reshape(len(x), 3))