import numpy as np


def tridiagonal(a, b, c=None):
    if c is None:
        c = a

    return np.diag(b) + np.diag(a, -1) + np.diag(c, 1)


def solve(
    V = None,
    I_ions = None,
    I_stim = None,
    Sv = None,
    C = None,
    dt = None,
    dx = None,
    dy = None,
    periodic = True
):
    if dy is not None:
        raise NotImplementedError("Solver hasn't been implemented for different delta x & y")

    nCols = V.shape[0]
    nRows = V.shape[1]
    if nCols != nRows:
        raise NotImplementedError("Solver hasn't been implemented for non-squared neuron sheets'")
    
    if not periodic:
        raise NotImplementedError("Solver hasn't been implemented for non-periodic neuron sheets")

    I = I_ions - I_stim
    lam = dt / (2 * Sv * dx ** (2) * C)
    M = (I * dt) / (2 * C)

    lambdas = np.full(nCols, lam)
    T = tridiagonal(lambdas, -(2 * lambdas + 1))

    if periodic:
        T[nRows - 1, 0] = lam
        T[0, nCols - 1] = lam

    TN = np.roll(V, (-1,0), (0,1)) # top neighbor
    BN = np.roll(V, (+1,0), (0,1)) # bottom neighbor

    Dx = (-lam) * TN + (2 * lam - 1) * V + (-lam) * BN
    V_n_half = np.zeros((nCols, nRows))

    for j in range(nRows):
        Dx_row = Dx[j,:]
        V_n_half[j,:] = np.linalg.solve(T, Dx_row)

    RN = np.roll(V_n_half, (0,-1), (0,1)) # right neighbor
    LN = np.roll(V_n_half, (0,+1), (0,1)) # left neighbor
    
    Dy = (-lam) * RN + (2 * lam - 1) * V_n_half + (-lam) * LN
    V_new = np.zeros((nCols, nRows))

    for i in range(nCols):
        Dy_row = Dy[:,i]
        V_new[:,i] = np.linalg.solve(T, Dy_row)

    return V_new
