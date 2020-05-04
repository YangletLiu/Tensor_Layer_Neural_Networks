import TensorAlgebra
import numpy as np
from scipy.linalg import solve
import tensorly as tl
import scipy as spy

def homo_encryption(original_tensor,key_Matrix,n):
    #初始化张量
    # tensor_1 = tl.tensor(np.arange(27).reshape(3, 3, 3).astype('float32'))
    # print(tensor_1)
    #加密矩阵
    # key_matrix = tl.tensor(np.arange(1,10).reshape(3, 3).astype('float32'))
    # print(key_matrix)

    encrypted_tensor = []
    for k in range(original_tensor.shape[0]):
        encrypted_factors = []
        for i in range(original_tensor.shape[2]):
            # X = key_matrix[i,0]*tensor_1[0,0,:] + key_matrix[i,1]*tensor_1[0,1,:] + key_matrix[i,2] * tensor_1[0,2,:]
            X = tl.tensor(np.zeros(n))
            for j in range(original_tensor.shape[1]):
                X = X + key_Matrix[i, j] * original_tensor[k, j, :]
            encrypted_factors.append(X)
        encrypted_factors = tl.fold(encrypted_factors, mode=0, shape=(n, n))
        # print(encrypted_factors)
        encrypted_tensor.append(encrypted_factors)
    encrypted_tensor = tl.fold(encrypted_tensor, mode=0, shape=(n, n, n))
    # print(encrypted_tensor)
    return encrypted_tensor

def homo_decryption(encrypted_Tensor,key_Matrix,n):
    decrypted_tensor = []
    for k in range(encrypted_Tensor.shape[0]):
        decrypted_columns = []
        for i in range(encrypted_Tensor.shape[2]):
            # x, info = cg(key_Matrix, encrypted_Tensor[k, :, i].reshape(n, 1))
            x = solve(key_Matrix,encrypted_Tensor[k, :, i])
            decrypted_columns.append(x)
        decrypted_factors = tl.transpose(tl.fold(decrypted_columns, mode=0, shape=(n, n)))
        # print(decrypted_factors)
        decrypted_tensor.append(decrypted_factors)
    decrypted_tensor = tl.fold(decrypted_tensor, mode=0, shape=(n, n, n))
    # print(decrypted_tensor)
    return decrypted_tensor

if __name__ == '__main__':
    factor_1 = tl.tensor(spy.sparse.rand(300, 5, 0.1, 'coo', dtype='float64').todense())
    factor_2 = tl.tensor(spy.sparse.rand(300, 5, 0.1, 'coo', dtype='float64').todense())
    factor_3 = tl.tensor(spy.sparse.rand(300, 5, 0.1, 'coo', dtype='float64').todense())

    factors_original = [factor_1, factor_2, factor_3]
    weight_original = np.array([1, 1, 1, 1, 1], dtype='float64')
    original_tensor = tl.kruskal_to_tensor([weight_original, factors_original])

    key_matrix = tl.tensor(np.random.random(90000).reshape(300, 300).astype('float64'))

    encrypted_tensor = homo_encryption(original_tensor, key_matrix, 300)

    cp_rank = 5
    factors,is_converge = TensorAlgebra.parafac(encrypted_tensor, cp_rank,n_iter_max=3000,init='svd', tol=10e-15)
    cp_reconstruction = TensorAlgebra.kruskal_to_tensor(factors)

    decrypted_tensor = homo_decryption(cp_reconstruction, key_matrix, 300)

    # loss
    delta = np.linalg.norm(original_tensor - decrypted_tensor)
    print('Loss:')
    print(delta)