import cupy as cp
import cupy.linalg as cpLA

def make_PSD(hess):
    # Eigen decomposition on symmetric matrix using CuPy
    lam, V = cpLA.eigh(hess)
    
    # Set all negative Eigenvalues to 0
    lam = cp.maximum(lam, 0)
    
    # Reconstruct the matrix
    return cp.matmul(cp.matmul(V, cp.diag(lam)), V.T)