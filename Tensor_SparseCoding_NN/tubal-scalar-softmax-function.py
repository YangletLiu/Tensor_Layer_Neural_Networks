import numpy as np
from tensor_product import t_product

#f_scalar_tube-wise function,which is implemented on every lateral slice of the tensor
def f_tube_wise(lateral_slice):
    #later slices
    # X_i = np.array(range(1,10)).reshape(3,3,1)
    l,m,n = lateral_slice.shape
    #tubes
    tubes =  [lateral_slice[:,i,0].reshape(l,1,1) for i in range(lateral_slice.shape[1])]
    tubes_sum = np.zeros(l).reshape(l,1,1)
    for i in tubes:
        # print(np.exp(i))
        tubes_sum += np.exp(i)
    tubes_sum = 1/tubes_sum
    # print(tubes_sum)
    h_tubes = [t_product(tubes_sum,i) for i in tubes]
    g_tubes = [i.sum() for i in h_tubes]
    # print(g_tubes)
    return g_tubes

#scalar_tubal_softmax function ,
#which is implemented on 3D-tensor to get the matrix of probabilities
def F_scalar_tubal_softmax(A_N):
    l,m,n = A_N.shape
    lateral_slices = [A_N[:,:,i] for i in range(A_N.shape[2])]
    X = []
    for tube in lateral_slices:
        tube = tube.reshape(l,m,1)
        res_tube = f_tube_wise(tube)
        # print(res_tube)
        X.append(res_tube)
    return np.vstack(X).transpose()

#tubal-scalar-softmax-example
A_N = np.array(range(1,9)).reshape(2,2,2)
print(F_scalar_tubal_softmax(A_N))