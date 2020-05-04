import numpy as np
import scipy.linalg
import scipy.sparse.linalg

def partial_svd(matrix, n_eigenvecs=None):
    # Check that matrix is... a matrix!
    if matrix.ndim != 2:
        raise ValueError('matrix be  a  matrix. matrix.ndim is {} != 2'.format(
            matrix.ndim))

    # Choose what to do depending on the params
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
    else:
        min_dim = dim_2

    if n_eigenvecs is None or n_eigenvecs >= min_dim:
        # Default on standard SVD
        U, S, V = scipy.linalg.svd(matrix)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
        return U, S, V

    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = scipy.sparse.linalg.eigsh(np.dot(matrix, matrix.T), k=n_eigenvecs, which='LM')

            S = np.sqrt(S)

            V = np.dot(matrix.T, U * 1/S[None, :])
        else:
            S, V = scipy.sparse.linalg.eigsh(np.dot(matrix.T, matrix), k=n_eigenvecs, which='LM')
            S = np.sqrt(S)
            U = np.dot(matrix, V) * 1/S[None, :]

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        return U, S, V.T
def norm(tensor, order):
    if order == 99:
        return np.max(np.abs(tensor))
    if order == 1:
        return np.sum(np.abs(tensor))
    elif order == 2:
        return np.sqrt(np.sum(tensor**2))
    else:
        return np.sum(np.abs(tensor)**order)**(1/order)


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def kr(matrices):
    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))
def khatri_rao(matrices, skip_matrix=None, reverse=False):
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    n_columns = matrices[0].shape[1]

    # Optional part, testing whether the matrices have the proper size
    for i, matrix in enumerate(matrices):
        if matrix.ndim != 2:
            raise ValueError('All the matrices must have exactly 2 dimensions!'
                             'Matrix {} has dimension {} != 2.'.format(
                                 i, matrix.ndim))
        if matrix.shape[1] != n_columns:
            raise ValueError('All matrices must have same number of columns!'
                             'Matrix {} has {} columns != {}.'.format(
                                 i, matrix.shape[1], n_columns))

    n_factors = len(matrices)

    if reverse:
        matrices = matrices[::-1]
        # Note: we do NOT use .reverse() which would reverse matrices even outside this function

    return kr(matrices)


def fold(unfolded_tensor, mode, shape):

    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)
def kruskal_to_tensor(factors):
    shape = [factor.shape[0] for factor in factors]
    full_tensor = np.dot(factors[0], khatri_rao(factors[1:]).T)
    return fold(full_tensor, 0, shape)
def parafac(tensor, rank, n_iter_max=100, init='svd', tol=10e-7,
            random_state=None, verbose=False):
    #print(tensor)
    plag=0
    if init is 'random':
        factors = [np.random.normal(size=(tensor.shape[i], rank))for i in range(tensor.ndim)]

    elif init is 'svd':
        factors = []
        for mode in range(tensor.ndim):
            U, _, _ = partial_svd(unfold(tensor, mode), n_eigenvecs=rank)

            if tensor.shape[mode] < rank:
                factor = np.array(np.zeros((U.shape[0], rank)))
                factor[:, tensor.shape[mode]:] = np.array(np.random.normal(size=(U.shape[0], rank - tensor.shape[mode])))#add shape.?
                factor[:, :tensor.shape[mode]] = U
                U = np.array(factor)
            factors.append(U[:, :rank])

    rec_errors = []
    norm_tensor = norm(tensor, 2)

    for iteration in range(n_iter_max):
        for mode in range(tensor.ndim):
            pseudo_inverse = np.array(np.ones((rank, rank)))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse[:] = pseudo_inverse*np.dot(factor.T, factor)
            factor = np.dot(unfold(tensor, mode), khatri_rao(factors, skip_matrix=mode))
            factor = np.linalg.solve(pseudo_inverse.T, factor.T).T
            factors[mode] = factor

        #if verbose or tol:
        rec_error = norm(tensor - kruskal_to_tensor(factors), 2) / norm_tensor
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print('reconsturction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                #if verbose:
                print('converged in {} iterations.'.format(iteration))
                plag=1
                break
    if not plag:
        print("not converged!!!!")
    not_converge = 0
    if not plag:
        print("not converged!!!!")
        not_converge = 1
    return factors,not_converge