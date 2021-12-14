import numpy as np
import _tdma

lambda_x = None
lambda_y = None

# From https://github.com/cpcloud/PyTDMA
def tdma(A, b):
    """Tridiagonal matrix solver.

    Parameters
    ----------
    A : M x N, array-like
    b : N, array-like

    Returns
    -------
    ret : N, array-like
    """
    lower = np.hstack((0, np.diag(A, -1)))
    middle = np.diag(A).copy()
    upper = np.hstack((np.diag(A, 1), 0))
    return np.asarray(_tdma.tdma(lower, middle, upper,
                                 np.asarray(b).squeeze()))


def tridiagonal(a, b, c=None):
    if c is None:
        c = a

    return np.diag(b) + np.diag(a, -1) + np.diag(c, 1)


def initialize_constants(
    Sv = None,
    C = None,
    dt = None,
    dx = None,
    dy = None
):
    global lambda_x, lambda_y

    lambda_x = dt / (2 * Sv * dx ** (2) * C)
    lambda_y = dt / (2 * Sv * dy ** (2) * C)

    
    

def solve(
    V = None,
    I_ions = None,
    I_stim = None,
    Sv = None,
    C = None,
    rhoDx = None,
    rhoDy = None,
    dt = None,
    dx = None,
    dy = None,
    periodicX = True,
    periodicY = True
):

    if lambda_x is None:
        initialize_constants(
            Sv = Sv,
            C = C,
            dt = dt,
            dx = dx,
            dy = dy
        )
        

    shape = V.shape
    I = I_ions - I_stim
    M = (I * dt) / (2 * C)

    # x-implicit step

    TN = np.roll(V, (+1,0), (0,1)) # top neighbor
    BN = np.roll(V, (-1,0), (0,1)) # bottom neighbor
    rhoDxRN = np.roll(rhoDx, (-1, 0))
    rhoDyBN = np.roll(rhoDy, (-1,0), (0,1))
    Dy = M + TN * (-lambda_y / rhoDy) + V * (lambda_y - 1 + (1 / rhoDy + 1 / rhoDyBN)) + BN * (-lambda_y / rhoDyBN)
    V_half = np.zeros(shape)

    rows = shape[1]
    for rowI in range(rows):
        a_rho = rhoDx[rowI,:]
        a = lambda_x / a_rho
        c_rho = rhoDxRN[rowI,:]
        c = lambda_x / c_rho
        b = -(1 + a + c)
        matrix = tridiagonal(a[1:], b, c[:-1])
        
        
        if periodicX:
            matrix[0, 0] = b[0] * 2
            matrix[rows - 1, rows - 1] = b[rows -1] + (a[0] * a[0]) / b[0] 
            u = np.zeros(rows)
            v = np.zeros(rows)
            u[0] = -b[0]
            u[rows - 1] = a[0]
            v[0] = 1
            v[rows - 1] = - a[0] / b[0]
            y = tdma(matrix, Dy[rowI,:])
            z = tdma(matrix, u)
            V_half[rowI,:] = y - ((v * y) / (1 + v * z)) * z
        else:
            V_half[rowI,:] = tdma(matrix, Dy[rowI,:])
    

    # y-implicit step

    RN = np.roll(V_half, (0,-1), (0,1)) # right neighbor
    LN = np.roll(V_half, (0,+1), (0,1)) # left neighbor
  
    Dx = M + LN * (-lambda_x / rhoDx) + V_half * (lambda_x - 1 + (1 / rhoDx + 1 / rhoDxRN)) + RN * (-lambda_x / rhoDxRN)
    V_new = np.zeros(shape)

    cols = shape[0]
    for colI in range(cols):
        a_rho = rhoDy[:,colI]
        a = lambda_y / a_rho
        c_rho = rhoDyBN[:,colI]
        c = lambda_y / c_rho
        b = -(1 + a + c)
        matrix = tridiagonal(a[1:], b, c[:-1])
        
        
        if periodicY:
            matrix[0, 0] = b[0] * 2
            matrix[rows - 1, rows - 1] = b[rows -1] + (a[0] * a[0]) / b[0] 
            u = np.zeros(rows)
            v = np.zeros(rows)
            u[0] = -b[0]
            u[rows - 1] = a[0]
            v[0] = 1
            v[rows - 1] = - a[0] / b[0]
            y = tdma(matrix, Dx[:,colI])
            z = tdma(matrix, u)
            V_half[:,colI] = y - ((v * y) / (1 + v * z)) * z
        else:
            V_new[:,colI] = tdma(matrix, Dx[rowI,:])

    return V_new

# def solve(
#     V = None,
#     I_ions = None,
#     I_stim = None,
#     Sv = None,
#     C = None,
#     Rho = None,
#     dt = None,
#     dx = None,
#     dy = None
# ):
#     global T_inv
#     if dy is not None:
#         raise NotImplementedError("Solver hasn't been implemented for different delta x & y")

#     nCols = V.shape[0]
#     nRows = V.shape[1]
#     if nCols != nRows:
#         raise NotImplementedError("Solver hasn't been implemented for non-squared neuron sheets'")

#     I = I_ions - I_stim
#     lam = dt / (2 * Sv * dx ** (2) * C * Rho)
#     M = (I * dt) / (2 * C)

#     # Inverse of tridiagonal-boundary matrix
#     if T_inv is None:
#         lambdas = np.full(nCols, lam)
#         T = tridiagonal(lambdas[:-1], -(2 * lambdas + 1))
#         T[nCols - 1, 0] = lam
#         T[0, nRows - 1] = lam
#         T_inv = np.linalg.inv(T)


#     TN = np.roll(V, (-1,0), (0,1)) # top neighbor
#     BN = np.roll(V, (+1,0), (0,1)) # bottom neighbor

#     Dx = (-lam) * TN + (2 * lam - 1) * V + (-lam) * BN + M
#     V_n_half = np.matmul(T_inv, Dx)

#     RN = np.roll(V_n_half, (0,-1), (0,1)) # right neighbor
#     LN = np.roll(V_n_half, (0,+1), (0,1)) # left neighbor
    
#     Dy = (-lam) * RN + (2 * lam - 1) * V_n_half + (-lam) * LN + M
#     V_new = np.matmul(T_inv, Dy.T) # TODO maybe it's not transpose

#     return V_new


    # I = I_ions - I_stim
    # lam = dt / (2 * Sv * dx ** (2) * C)
    # M = (I * dt) / (2 * C)

    # lambdas = np.full(nCols, lam)
    # T = tridiagonal(lambdas[:-1], -(2 * lambdas + 1))

    # b1 = -(2 * lam + 1)
    # cn = lam
    # a1 = lam

    # # Correct for u and v
    # T[0, 0] = T[0, 0] + b1
    # T[nRows - 1, nRows - 1] = T[nRows - 1, nRows - 1] + (a1 ** (2) / b1)

    # # Construct u
    # u = np.zeros(nRows)
    # u[0] = -b1
    # u[nRows - 1] = cn

    # # Inverse of tridiagonal-boundarymatrix
    # T_inv = None

    # # Construct v
    # v = np.zeros(nRows)
    # v[0] = 1
    # v[nRows - 1] = - a1 / b1

    # q = tdma(T, u)

    # TN = np.roll(V, (-1,0), (0,1)) # top neighbor
    # BN = np.roll(V, (+1,0), (0,1)) # bottom neighbor

    # Dx = (-lam) * TN + (2 * lam - 1) * V + (-lam) * BN + M
    # V_n_half = np.zeros((nCols, nRows))

    # for j in range(nRows):
    #     Dx_row = Dx[j,:]
    #     y = tdma(T, Dx_row)
    #     #V_n_half[j,:] = y - (np.dot(v, y) / (1 + np.dot(v, q))) * q # TODO is the np.dot correct?
    #     V_n_half[j,:] = y

    # RN = np.roll(V_n_half, (0,-1), (0,1)) # right neighbor
    # LN = np.roll(V_n_half, (0,+1), (0,1)) # left neighbor
    
    # Dy = (-lam) * RN + (2 * lam - 1) * V_n_half + (-lam) * LN + M
    # V_new = np.zeros((nCols, nRows))

    # for i in range(nCols):
    #     Dy_row = Dy[:,i]
    #     y = tdma(T, Dy_row)
    #     #V_new[:,i] = y - (np.dot(v, y) / (1 + np.dot(v, q))) * q
    #     V_new[:,i] = y

    # return V_new
